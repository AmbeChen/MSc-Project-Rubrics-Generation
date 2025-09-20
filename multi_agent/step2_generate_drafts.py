#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Generate rubrics from conversation files + RAG JSONL ("passages") + few-shot.

Requirements:
- Evidence: read ONLY from 'passages' in the RAG JSONL; keep each passage block labeled by its 'id' and
  concatenate all passages for the prompt. evidence_id must come from those ids.
- Conversation: load from --conversation_dir using --conversation_template (e.g., conversation_{index}.txt).
- Axes: must be one of the fixed five, no heuristic mapping beyond simple string normalization:
    ["accuracy","completeness","context_awareness","communication_quality","instruction_following"]
- Points: integer in [-10, 10]; positive = reward; negative = penalty; 0 = neutral.
- Outputs per index: pretty JSON, readable Markdown, and prompt text; plus a summary JSONL (one line per index).
- Supports batching with --start and --end.

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
# Fixed axes (lowercase with underscores)
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

# --------------------------
# Conversation (simple direct read)
# --------------------------

def read_conversation_from_file(directory: str, template: str, index: int, encoding: str = "utf-8") -> List[Dict[str, str]]:
    """
    Load conversation file and parse role-tagged lines if present.
    Accepts 'User:', 'Assistant:', 'System:' prefixes (case-insensitive). If none found, full text -> single user turn.
    """
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
# Evidence (ONLY from 'passages')
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
    """
    Expect structure like:
    {
      "index": 0,
      "queries": [...],
      "passages": [
        {"source": "...", "url": "...", "text": "...", "id": "p1"},
        ...
      ]
    }
    """
    val = example.get("passages", None)
    if isinstance(val, list) and val and isinstance(val[0], dict):
        return val
    return []

def build_evidence_from_passages(passages: List[Dict[str, Any]], max_chars_per_doc: int = 1200) -> Tuple[str, List[str]]:
    """
    Keep each passage as a separate block labeled by its 'id' and concatenate all.
    Return the evidence text + the list of valid evidence ids.
    """
    if not passages:
        return "[EV_0] No external evidence provided.\n(No URL)\n", ["EV_0"]

    blocks = []
    ids: List[str] = []
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
# Few-shot (kept simple)
# --------------------------

def build_fewshot_block(fewshots: List[Dict[str, Any]], k: int = 2) -> str:
    """
    Format a small number of exemplars. We do not map axes; we keep them as-is
    but they should already use the allowed set in your reference data.
    """
    exemplars = []
    used = 0
    for fs in fewshots:
        if used >= k:
            break
        conv = fs.get("conversation") or fs.get("messages") or fs.get("turns") or []
        rub = fs.get("rubrics") or fs.get("reference_rubrics") or fs.get("ref_rubrics") or []
        if not isinstance(conv, list) or not isinstance(rub, list):
            continue

        conv_text = "\n".join([f"{t.get('role','user')}: {t.get('content','')}" for t in conv])

        # Keep items but pretty-print so the schema is visible
        exemplars.append(
            "### Example Conversation\n" + conv_text + "\n" +
            "### Example Rubrics (JSON)\n" + json.dumps(rub, ensure_ascii=False, indent=2)
        )
        used += 1

    return "\n\n".join(exemplars) if exemplars else "(No exemplars)"

# --------------------------
# Prompting, parsing, normalization
# --------------------------
# --------------------------
# Prompting (focused on criterion quality, uniqueness, and optional evidence)
# --------------------------

AXES_SET = "['accuracy','completeness','context_awareness','communication_quality','instruction_following']"

SYSTEM_PROMPT = (
        "You are a medical assistant tasked with evaluating model responses in medical conversations.\n"
        "You will be given EXAMPLES of how to generate rubrics. Then, you will be asked to generate rubrics for a NEW conversation.\n\n"
        "Each rubric should:\n"
        "- contain a clear evaluation criterion (what to look for)\n"
        "- specify an axis: one of completeness, accuracy, context_awareness, communication_quality, instruction_following\n"
        "- assign a point between -10 and 10 (positive for good behavior, negative for harmful/incomplete info)\n"
        "- evidence_id: choose a passage id **only if** it clearly supports this item; otherwise use 'EV_0'.\n\n"
)

USER_PROMPT_TEMPLATE = (
    "=== FEW-SHOT EXAMPLES ===\n"
    "{fewshot_block}\n"
    "\n"
    "### TARGET CONVERSATION\n"
    "{conversation_block}\n"
    "\n"
    "### Reference Information / Evidence (id -> content)\n"
    "{evidence_block}\n"
    "Note: Using a passage is optional. If none is a good fit, set evidence_id='EV_0'.\n"
    "\n"
    "### Task\n"
    "Please ensure the following when generating rubrics:\n\n"
    "- Generate **10 distinct criteria**, each rubric must cover **completeness, accuracy, and context_awareness** axes.\n"
    "- Include both **positive** and **negative** criteria.\n"
    "   - Positive rubrics: describe correct, helpful, or exemplary assistant behaviors (assign positive point values).\n"
    "   - Negative rubrics: describe missing, incorrect, misleading, harmful, or otherwise poor behaviors (assign negative point values, e.g., -1 to -10).\n"
    "- Each rubric must be **directly related to the specific conversation content**. Do not include generic or unrelated criteria.\n"
    "- Try to cover all five axes: completeness, accuracy, context_awareness, communication_quality, instruction_following\n"
    "\n"
    
    "Now generate rubrics in JSON format as a list. Each item should include:\n"
        "- criterion (string)\n"
        "- axis (completeness | accuracy | context_awareness | communication_quality | instruction following)\n"
        "- point (integer between -10 and 10)\n"
        "- evidence_id (string)\n\n"
        "Rubrics:\n"
)

def parse_json_safely(text: str):
    """
    Robustly parse model output as a JSON-like structure.
    Accepts:
      - Valid JSON
      - JSON fenced in ```json
      - Python-literal style with single quotes (via ast.literal_eval)
      - Slightly malformed arrays missing trailing bracket
    Returns a Python list/dict or None.
    """
    # Normalize “smart quotes” to ASCII, which often sneak in
    text = (text
            .replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'"))

    # 1) fenced code block first
    m = re.search(r"```json\s*(\[.*?\]|\{.*?\})\s*```", text, flags=re.S)
    if m:
        snippet = m.group(1)
        # try strict JSON, then Python literal
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(snippet)
                # if a single dict is returned, normalize to list
                if isinstance(obj, dict):
                    return [obj]
                return obj
            except Exception:
                pass

    # 2) try strict parse of the largest bracketed slice
    start = min([i for i in (text.find("["), text.find("{")) if i != -1], default=-1)
    end = max(text.rfind("]"), text.rfind("}"))
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1].strip()
        # try strict JSON, then Python literal
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(snippet)
                if isinstance(obj, dict):
                    return [obj]
                return obj
            except Exception:
                pass

    # 3) balanced-brace extraction: collect all top-level objects, using ast for single-quoted dicts
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
        return objs  # array of parsed dicts

    # 4) last resort: coerce missing closing bracket for arrays like "[{...},{...}"
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
    """
    Only normalize formatting: lowercase and replace spaces/hyphens with underscores.
    Reject anything outside the allowed set.
    """
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
        # Support "axis" or "axis" in tags
        axis_raw = it.get("axis", "")
        if not axis_raw and "tags" in it and isinstance(it["tags"], list):
            axis_raw = ""
            for tag in it["tags"]:
                m = re.match(r"axis:(\w+)", tag)
                if m:
                    axis_raw = m.group(1)
                    break
        axis = normalize_axis_str(axis_raw)
        # Support "point" or "points"
        point = it.get("point", it.get("points", 0))
        try:
            point = int(point)
        except Exception:
            point = 0
        # clamp to [-10, 10]
        if point < -10: point = -10
        if point > 10:  point = 10
        evid = str(it.get("evidence_id", fallback)).strip() or fallback
        if valid_evid_ids and evid not in valid_evid_ids:
            evid = fallback

        if criterion and axis is not None:
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
# Human-readable draft
# --------------------------

def save_readable_markdown(path: str, conversation_block: str, evidence_block: str, rubrics: List[Dict[str, Any]]) -> None:
    lines = []
    lines.append("# Rubrics Draft\n")
    lines.append("## Conversation (preview)\n")
    lines.append("```\n" + (conversation_block[:2000] if conversation_block else "") + "\n```\n")
    lines.append("## Evidence\n")
    lines.append("```\n" + (evidence_block[:4000] if evidence_block else "") + "\n```\n")
    lines.append("## Rubrics\n")
    if not rubrics:
        lines.append("_No rubric items parsed._\n")
    else:
        for idx, r in enumerate(rubrics, 1):
            lines.append(f"### Item {idx}")
            lines.append(f"- **axis:** {r.get('axis','')}")
            lines.append(f"- **point:** {r.get('point','')}")
            lines.append(f"- **evidence_id:** {r.get('evidence_id','')}")
            lines.append(f"- **criterion:** {r.get('criterion','')}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def load_summary_jsonl(path: str) -> Dict[int, Any]:
    """Read the existing summary_jsonl and return the mapping from index to rubrics."""
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
    ap.add_argument("--fewshot_jsonl", type=str, required=True)
    ap.add_argument("--conversation_dir", type=str, required=True)
    ap.add_argument("--conversation_template", type=str, default="conversation_{index}.txt")
    ap.add_argument("--conversation_encoding", type=str, default="utf-8")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--summary_jsonl", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--fewshot_k", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--max_new_tokens", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.7)  # deterministic by default
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.summary_jsonl) or ".")

    rag_data = read_jsonl(args.rag_jsonl)
    fewshot_data = read_jsonl(args.fewshot_jsonl)
    fewshot_block = build_fewshot_block(fewshot_data, k=args.fewshot_k)

    tokenizer, model, pipe = load_model_and_tokenizer(args.model_name)

    if args.debug:
        print(f"[DEBUG] loaded {len(rag_data)} RAG examples; {len(fewshot_data)} few-shot examples", file=sys.stderr)

    # 1. Read the existing summary
    summary_map = load_summary_jsonl(args.summary_jsonl)

    with open(args.summary_jsonl, "a", encoding="utf-8") as sum_f:  # Use the append mode
        for i, ex in enumerate(rag_data):
            ex_idx = extract_index(ex, fallback=i)
            if ex_idx < args.start or ex_idx >= args.end:
                continue

            # Conversation
            conversation = read_conversation_from_file(
                args.conversation_dir, args.conversation_template, ex_idx, args.conversation_encoding
            )
            conversation_block = build_conversation_text(conversation) if conversation else "(No conversation provided)"

            # Evidence strictly from 'passages'
            passages = extract_passages(ex)
            evidence_block, valid_evid_ids = build_evidence_from_passages(passages)

            if args.debug:
                print(
                    f"[DEBUG] index={ex_idx} turns={len(conversation)} passages={len(passages)} evid_ids={len(valid_evid_ids)}",
                    file=sys.stderr
                )

            # Prompt
            user_prompt = USER_PROMPT_TEMPLATE.format(
                fewshot_block=fewshot_block,
                evidence_block=evidence_block,
                conversation_block=conversation_block,
            )
            messages = build_messages(SYSTEM_PROMPT, user_prompt)
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
            rubrics = normalize_rubrics(parsed, valid_evid_ids=valid_evid_ids)

            # Per-index outputs
            per = {
                "index": ex_idx,
                "conversation_preview": conversation_block[:500] if isinstance(conversation_block, str) else "",
                "rubrics": rubrics,
                "raw_model_output": raw_text,
            }
            json_path = os.path.join(args.out_dir, f"{ex_idx}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(per, f, ensure_ascii=False, indent=2)

            md_path = os.path.join(args.out_dir, f"{ex_idx}.md")
            save_readable_markdown(md_path, conversation_block, evidence_block, rubrics)

            prompt_path = os.path.join(args.out_dir, f"{ex_idx}.prompt.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)

            # 2. Update summary_map (overwrite existing index)
            summary_map[ex_idx] = {"index": ex_idx, "rubrics": rubrics}

            print(f"[OK] index={ex_idx} -> {json_path}  (also wrote {md_path})")

    # 3. Write back summary_jsonl (overwrite old file)
    with open(args.summary_jsonl, "w", encoding="utf-8") as sum_f:
        for obj in sorted(summary_map.values(), key=lambda x: x["index"]):
            sum_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
