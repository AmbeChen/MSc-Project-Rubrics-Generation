#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: Review rubrics (multi-agent 'reviewing agent').

Goals:
- Input: step2 drafts (12 items), conversation text, and RAG JSONL passages (evidence).
- Model: Llama-3-8B-Instruct, same as in step2.
- Tasks for reviewing agent:
  * Detect and remove redundancy/overlap (at most 2 deletions from original).
  * Fill missing medically substantive checks (use evidence when clearly supportive).
  * Keep EXACTLY 12 final items after review.
  * Each item: {criterion, axis, point, evidence_id}
    - axis ∈ ["accuracy","completeness","context_awareness","communication_quality","instruction_following"]
    - point ∈ [-10,10], integer, non-zero
    - evidence_id must be from the provided passages' ids, else 'EV_0'
  * Non-questions; behavior-based, concrete, single-facet, non-overlapping.

Hard constraint enforcement post-process:
- Ensure final list length == 12.
- Ensure at most two deletions of original criteria:
  -> At least 10 original criteria must remain.
  -> If model preserved <10, automatically reinsert originals (preferring non-conflicting) to reach 10.
  -> Then trim/compose to exactly 12.

Outputs per index:
- {out_dir}/{index}.json  (final rubrics + bookkeeping)
- {out_dir}/{index}.md    (human-readable preview)
- {out_dir}/{index}.prompt.txt (actual prompt sent to model)
- Appends/rewrites a summary_jsonl with {"index": idx, "rubrics": [...]} lines.

Code is English-only by request.
"""

import argparse
import json
import os
import re
import sys
import ast
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --------------------------
# Fixed axes
# --------------------------

ALLOWED_AXES = [
    "accuracy",
    "completeness",
    "context_awareness",
    "communication_quality",
    "instruction_following",
]

# --------------------------
# IO helpers
# --------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Skipping malformed line in {path}: {e}", file=sys.stderr)
    return data

def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

def read_text(path: str, encoding: str = "utf-8") -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding=encoding) as f:
        return f.read()

# --------------------------
# Conversation utilities (copy compatible with step2)
# --------------------------

def read_conversation_from_file(directory: str, template: str, index: int, encoding: str = "utf-8") -> List[Dict[str, str]]:
    filename = template.format(index=index)
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        print(f"[WARN] Conversation file not found: {path}", file=sys.stderr)
        return []
    with open(path, "r", encoding=encoding) as f:
        text = f.read()

    lines = text.splitlines()
    convo: List[Dict[str, str]] = []
    cur_role = None
    buf: List[str] = []

    def flush():
        nonlocal convo, cur_role, buf
        if buf:
            role = cur_role if cur_role else "user"
            content = "\n".join(buf).strip()
            if content:
                convo.append({"role": role, "content": content})
        cur_role, buf = None, []

    for line in lines:
        m = re.match(r"^\s*(user|assistant|system)\s*:\s*(.*)$", line, flags=re.I)
        if m:
            flush()
            cur_role = m.group(1).lower()
            buf = [m.group(2)]
        else:
            buf.append(line)
    flush()

    if not convo:
        text = text.strip()
        if text:
            convo = [{"role": "user", "content": text}]
    return convo

def build_conversation_text(conversation: List[Dict[str, str]], max_chars: int = 8000) -> str:
    lines = [f"{t.get('role','user')}: {t.get('content','')}" for t in conversation]
    joined = "\n".join(lines)
    if len(joined) > max_chars:
        joined = joined[-max_chars:]
    return joined

# --------------------------
# Evidence (ONLY from 'passages' in rag_jsonl)
# --------------------------

def extract_index(example: Dict[str, Any], fallback: int) -> int:
    for k in ["index", "idx", "id"]:
        if k in example:
            try:
                return int(example[k])
            except Exception:
                pass
    return fallback

def extract_passages(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    val = example.get("passages", None)
    if isinstance(val, list) and val and isinstance(val[0], dict):
        return val
    return []

def build_evidence_from_passages(passages: List[Dict[str, Any]], max_chars_per_doc: int = 1200) -> Tuple[str, List[str]]:
    if not passages:
        return "[EV_0] No external evidence provided.\n(No URL)\n", ["EV_0"]
    blocks, ids = [], []
    for i, doc in enumerate(passages):
        evid = str(doc.get("id", f"EV_{i}"))
        title = str(doc.get("title", doc.get("source", "")))
        body = ""
        for k in ["text", "snippet", "content", "summary", "abstract"]:
            if k in doc and doc[k]:
                body = str(doc[k])
                break
        if len(body) > max_chars_per_doc:
            body = body[:max_chars_per_doc] + " ..."
        url = str(doc.get("url", ""))
        block = f"[{evid}] {title}\n{body}\n{url}".strip()
        blocks.append(block)
        ids.append(evid)
    return "\n\n".join(blocks) + "\n", ids

# --------------------------
# Drafts from step2
# --------------------------

def read_step2_draft(drafts_dir: str, index: int) -> Dict[str, Any]:
    path = os.path.join(drafts_dir, f"{index}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --------------------------
# Few-shot (optional; illustrative only)
# --------------------------

def build_fewshot_block(fewshots: List[Dict[str, Any]], k: int = 1) -> str:
    exemplars, used = [], 0
    for fs in fewshots:
        if used >= k:
            break
        conv = fs.get("conversation") or fs.get("messages") or fs.get("turns") or []
        rub = fs.get("rubrics") or fs.get("reference_rubrics") or fs.get("ref_rubrics") or []
        if not isinstance(conv, list) or not isinstance(rub, list):
            continue
        conv_text = "\n".join([f"{t.get('role','user')}: {t.get('content','')}" for t in conv])
        exemplars.append(
            "### Example Conversation\n" + conv_text + "\n" +
            "### Example Rubrics (JSON)\n" + json.dumps(rub, ensure_ascii=False, indent=2)
        )
        used += 1
    return "\n\n".join(exemplars) if exemplars else "(No exemplars)"

# --------------------------
# Parsing / normalization
# --------------------------

def parse_json_safely(text: str):
    text = (text
            .replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'"))

    m = re.search(r"```json\s*(\[.*?\]|\{.*?\})\s*```", text, flags=re.S)
    if m:
        snippet = m.group(1)
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(snippet)
                if isinstance(obj, dict):
                    return [obj]
                return obj
            except Exception:
                pass

    start = min([i for i in (text.find("["), text.find("{")) if i != -1], default=-1)
    end = max(text.rfind("]"), text.rfind("}"))
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1].strip()
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(snippet)
                if isinstance(obj, dict):
                    return [obj]
                return obj
            except Exception:
                pass

    objs = []
    depth = 0
    obj_start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start != -1:
                    cand = text[obj_start:i+1]
                    for parser in (json.loads, ast.literal_eval):
                        try:
                            obj = parser(cand)
                            if isinstance(obj, dict):
                                objs.append(obj)
                                break
                        except Exception:
                            continue
                    obj_start = -1
    if objs:
        return objs

    stripped = text.strip()
    if stripped.startswith("[") and not stripped.endswith("]"):
        try:
            obj = json.loads(stripped + "]")
            return obj
        except Exception:
            try:
                obj = ast.literal_eval(stripped + "]")
                if isinstance(obj, dict):
                    return [obj]
                return obj
            except Exception:
                pass

    return None

def normalize_axis_str(axis: str) -> Optional[str]:
    if axis is None:
        return None
    a = axis.strip().lower().replace(" ", "_").replace("-", "_")
    return a if a in ALLOWED_AXES else None

def normalize_rubrics(items: Any, valid_evid_ids: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out = []
    fallback = valid_evid_ids[0] if valid_evid_ids else "EV_0"
    for it in items:
        if not isinstance(it, dict):
            continue
        criterion = str(it.get("criterion", "")).strip()
        axis_raw = it.get("axis", "")
        axis = normalize_axis_str(axis_raw)
        point = it.get("point", 0)
        try:
            point = int(point)
        except Exception:
            point = 0
        point = max(-10, min(10, point))
        evid = str(it.get("evidence_id", fallback)).strip() or fallback
        if valid_evid_ids and evid not in valid_evid_ids:
            evid = fallback
        if criterion and axis is not None and point != 0:
            out.append({
                "criterion": criterion,
                "axis": axis,
                "point": point,
                "evidence_id": evid
            })
    # de-duplicate by (criterion, axis, evidence_id)
    seen, dedup = set(), []
    for it in out:
        key = (it["criterion"], it["axis"], it["evidence_id"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)
    return dedup

# --------------------------
# Prompting
# --------------------------

SYSTEM_REVIEW_PROMPT = (
    "You are a senior medical rubric editor. You will REVIEW a 12-item draft rubric and produce a FINAL 12-item rubric.\n"
    "- Remove redundancy and overlaps but delete AT MOST TWO original items.\n"
    "- Add new medically substantive items (if needed) so the final count remains EXACTLY 12.\n"
    "- Use the provided evidence passages when they clearly support an item; otherwise use 'EV_0'.\n"
    "- Focus on concrete, single-behavior, decisionable checks; no questions; avoid superficial rewording.\n"
    "- axis ∈ ['accuracy','completeness','context_awareness','communication_quality','instruction_following'].\n"
    "- point is INTEGER in [-10,10] and NON-ZERO (use larger magnitude for safety/factual risks).\n"
    "Return ONLY a JSON array of 12 objects with keys: criterion, axis, point, evidence_id."
)

USER_REVIEW_TEMPLATE = (
    "### Task\n"
    "Review the DRAFT rubric below. Remove at most TWO items that are redundant/overlapping or low-value, then ADD replacements to address missing but critical checks suggested by the conversation and the evidence. The FINAL must contain EXACTLY 12 items.\n"
    "\n"
    "Behavior/form requirements per item:\n"
    "- criterion: behavior-based, specific, verifiable in the unseen reply; not a question; avoid vague/meta phrasing.\n"
    "- axis: choose exactly ONE from ['accuracy','completeness','context_awareness','communication_quality','instruction_following'].\n"
    "- point: integer in [-10,10], NON-ZERO; higher magnitude for safety/factual-critical.\n"
    "- evidence_id: use a passage id only when clearly supportive; otherwise 'EV_0'.\n"
    "\n"
    "### Few-shot (illustrative, do NOT copy wording)\n"
    "{fewshot_block}\n"
    "\n"
    "### Evidence (id -> content)\n"
    "{evidence_block}\n"
    "Note: using a passage is optional; never invent ids.\n"
    "\n"
    "### Conversation\n"
    "{conversation_block}\n"
    "\n"
    "### DRAFT Rubrics (JSON; 12 items)\n"
    "{draft_block}\n"
    "\n"
    "### Output\n"
    "Return ONLY the FINAL 12-item JSON array, no commentary."
)

def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = ""
    for m in messages:
        text += f"{m['role'].upper()}: {m['content']}\n"
    text += "ASSISTANT:"
    return text

# --------------------------
# Model loader
# --------------------------

def load_model_and_tokenizer(model_name: str, dtype: str = "bfloat16"):
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" and torch.cuda.is_available() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    return tokenizer, model, pipe

# --------------------------
# Utility: enforce at-most-two deletions
# --------------------------

def enforce_max_two_deletions(original: List[Dict[str, Any]], reviewed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure at least 10 original criteria are preserved (=> at most two deletions).
    If fewer preserved, reinsert original items until preserved==10, then trim/compose to 12.
    Matching is done by exact criterion string equality.
    """
    orig_by_criterion = {it["criterion"]: it for it in original}
    reviewed_by_criterion = {it["criterion"]: it for it in reviewed}

    preserved = [c for c in reviewed if c["criterion"] in orig_by_criterion]
    preserved_set = set([it["criterion"] for it in preserved])

    # If preserved < 10, add back originals not present
    if len(preserved) < 10:
        need = 10 - len(preserved)
        for it in original:
            if need <= 0:
                break
            if it["criterion"] not in preserved_set:
                reviewed.append(it)
                preserved_set.add(it["criterion"])
                need -= 1

    # Deduplicate again by (criterion, axis, evidence_id)
    seen, dedup = set(), []
    for it in reviewed:
        key = (it["criterion"], it["axis"], it["evidence_id"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)

    # Ensure length == 12
    if len(dedup) > 12:
        dedup = dedup[:12]
    elif len(dedup) < 12:
        # If still short (e.g., due to strict normalization removing items), pad with remaining originals
        for it in original:
            key = (it["criterion"], it["axis"], it["evidence_id"])
            if key in seen:
                continue
            dedup.append(it)
            seen.add(key)
            if len(dedup) >= 12:
                break

    # Final clamp
    return dedup[:12]

# --------------------------
# Human-readable save
# --------------------------

def save_readable_markdown(path: str, conversation_block: str, evidence_block: str, draft: List[Dict[str, Any]], final_items: List[Dict[str, Any]]) -> None:
    lines = []
    lines.append("# Rubrics Review\n")
    lines.append("## Conversation (preview)\n")
    lines.append("```\n" + (conversation_block[:2000] if conversation_block else "") + "\n```\n")
    lines.append("## Evidence\n")
    lines.append("```\n" + (evidence_block[:4000] if evidence_block else "") + "\n```\n")
    lines.append("## Draft (from Step 2)\n")
    if not draft:
        lines.append("_No draft items loaded._\n")
    else:
        for idx, r in enumerate(draft, 1):
            lines.append(f"### Draft Item {idx}")
            lines.append(f"- **axis:** {r.get('axis','')}")
            lines.append(f"- **point:** {r.get('point','')}")
            lines.append(f"- **evidence_id:** {r.get('evidence_id','')}")
            lines.append(f"- **criterion:** {r.get('criterion','')}\n")
    lines.append("## FINAL (Step 3 Reviewed)\n")
    if not final_items:
        lines.append("_No final items parsed._\n")
    else:
        for idx, r in enumerate(final_items, 1):
            lines.append(f"### Final Item {idx}")
            lines.append(f"- **axis:** {r.get('axis','')}")
            lines.append(f"- **point:** {r.get('point','')}")
            lines.append(f"- **evidence_id:** {r.get('evidence_id','')}")
            lines.append(f"- **criterion:** {r.get('criterion','')}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def load_summary_jsonl(path: str) -> Dict[int, Any]:
    summary = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    idx = int(obj.get("index", -1))
                    if idx >= 0:
                        summary[idx] = obj
                except Exception:
                    continue
    return summary

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_jsonl", type=str, required=True)
    ap.add_argument("--conversation_dir", type=str, required=True)
    ap.add_argument("--conversation_template", type=str, default="conversation_{index}.txt")
    ap.add_argument("--conversation_encoding", type=str, default="utf-8")

    ap.add_argument("--drafts_dir", type=str, required=True, help="Directory of step2 outputs (contains {index}.json).")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--summary_jsonl", type=str, required=True)

    ap.add_argument("--fewshot_jsonl", type=str, required=False, default="")
    ap.add_argument("--fewshot_k", type=int, default=1)

    ap.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.summary_jsonl) or ".")

    rag_data = read_jsonl(args.rag_jsonl)
    fewshot_data = read_jsonl(args.fewshot_jsonl) if args.fewshot_jsonl else []
    fewshot_block = build_fewshot_block(fewshot_data, k=args.fewshot_k)

    tokenizer, model, pipe = load_model_and_tokenizer(args.model_name)

    # load previous summary (if any)
    summary_map = load_summary_jsonl(args.summary_jsonl)

    for i, ex in enumerate(rag_data):
        ex_idx = extract_index(ex, fallback=i)
        if ex_idx < args.start or ex_idx >= args.end:
            continue

        # Conversation
        conversation = read_conversation_from_file(
            args.conversation_dir, args.conversation_template, ex_idx, args.conversation_encoding
        )
        conversation_block = build_conversation_text(conversation) if conversation else "(No conversation provided)"

        # Evidence
        passages = extract_passages(ex)
        evidence_block, valid_evid_ids = build_evidence_from_passages(passages)

        # Step2 draft
        draft_obj = read_step2_draft(args.drafts_dir, ex_idx)
        draft_items = draft_obj.get("rubrics", []) if isinstance(draft_obj, dict) else []
        if not draft_items:
            print(f"[WARN] No draft items for index={ex_idx}; skipping.", file=sys.stderr)
            continue

        # Prepare prompt
        draft_block = json.dumps(draft_items, ensure_ascii=False, indent=2)
        user_prompt = USER_REVIEW_TEMPLATE.format(
            fewshot_block=fewshot_block,
            evidence_block=evidence_block,
            conversation_block=conversation_block,
            draft_block=draft_block,
        )
        messages = build_messages(SYSTEM_REVIEW_PROMPT, user_prompt)
        prompt_text = apply_chat_template(tokenizer, messages)

        # Generate
        gen = pipe(
            prompt_text,
            max_new_tokens=args.max_new_tokens,
            do_sample=(args.temperature > 0.0),
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        raw_text = gen[0]["generated_text"] if gen else ""

        # Parse + normalize
        parsed = parse_json_safely(raw_text)
        reviewed_items = normalize_rubrics(parsed, valid_evid_ids=valid_evid_ids)

        # Fallback if model failed: start from draft_items
        if not reviewed_items:
            reviewed_items = normalize_rubrics(draft_items, valid_evid_ids=valid_evid_ids)

        # Enforce hard constraints: exactly 12; at-most-two deletions
        original_norm = normalize_rubrics(draft_items, valid_evid_ids=valid_evid_ids)
        final_items = enforce_max_two_deletions(original_norm, reviewed_items)

        # Per-index outputs
        per = {
            "index": ex_idx,
            "conversation_preview": conversation_block[:500] if isinstance(conversation_block, str) else "",
            "draft_count": len(draft_items),
            "final_count": len(final_items),
            "rubrics": final_items,
            "raw_model_output": raw_text,
        }
        json_path = os.path.join(args.out_dir, f"{ex_idx}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(per, f, ensure_ascii=False, indent=2)

        md_path = os.path.join(args.out_dir, f"{ex_idx}.md")
        save_readable_markdown(md_path, conversation_block, evidence_block, original_norm, final_items)

        prompt_path = os.path.join(args.out_dir, f"{ex_idx}.prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)

        # Update summary (overwrite existing index)
        summary_map[ex_idx] = {"index": ex_idx, "rubrics": final_items}

        print(f"[OK] reviewed index={ex_idx} -> {json_path}  (also wrote {md_path})")

    # Write back summary_jsonl
    with open(args.summary_jsonl, "w", encoding="utf-8") as sum_f:
        for obj in sorted(summary_map.values(), key=lambda x: x["index"]):
            sum_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
