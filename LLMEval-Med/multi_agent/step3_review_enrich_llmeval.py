#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3 (LLMEval-Med): Review, merge and optionally enrich rubrics (checklists).

What this does:
  1) KEEP all items that are relevant and non-duplicative.
  2) MERGE semantically similar but complementary items into ONE more specific checklist line.
  3) REMOVE overly generic/boilerplate items.
  4) (Optional) ADD up to N missing-but-important items IF strongly supported by evidence passages.

Inputs:
  - --rag_jsonl        : step1.5 output (contains index, prompt (role/content), passages[id,url,text/title...])
  - --draft_jsonl      : step2 summary (contains index, rubrics=[{"criterion": "...", "evidence_id": "..." }])
Outputs:
  - per-index JSON     : { index, kept: [{criterion, evidence_id}], removed: [...], merged: [...], added: [...] }
  - summary JSONL      : { index, rubrics: [{criterion, evidence_id}] } one line per index

Model:
  - meta-llama/Meta-Llama-3-8B-Instruct (默认)
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --------------------------
# IO helpers
# --------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] skip malformed line in {path}: {e}", file=sys.stderr)
    return out

def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

# --------------------------
# Conversation / Evidence
# --------------------------

def build_conversation_text_from_prompt(prompt: List[Dict[str, str]], max_chars: int = 8000) -> str:
    if not isinstance(prompt, list):
        return ""
    lines = [f"{t.get('role','user')}: {t.get('content','')}" for t in prompt]
    joined = "\n".join(lines)
    if len(joined) > max_chars:
        joined = joined[-max_chars:]
    return joined

def build_evidence_block(passages: List[Dict[str, Any]], max_chars_per_doc: int = 1000) -> Tuple[str, List[str]]:
    """
    Build labeled evidence blocks and return (evidence_text, valid_ids).
    """
    if not passages:
        return "[EV_0] No external evidence provided.\n", ["EV_0"]
    blocks = []
    ids = []
    for i, p in enumerate(passages):
        evid = str(p.get("id", f"p{i+1}"))
        title = str(p.get("title") or p.get("source") or "")
        body  = str(p.get("text") or p.get("snippet") or "")
        url   = str(p.get("url") or "")
        if len(body) > max_chars_per_doc:
            body = body[:max_chars_per_doc] + " ..."
        blocks.append(f"[{evid}] {title}\n{body}\n{url}".strip())
        ids.append(evid)
    return "\n\n".join(blocks) + "\n", ids

# --------------------------
# Simple text utilities
# --------------------------

GENERIC_PATTERNS = [
    r"^回答是否.*$", r"^是否.*$", r"^需不需要.*$", r"^有没有.*$", r"^注意.*$", r"^加强.*$",
    r"^强调.*$", r"^建议.*$", r"^提供.*$", r"^包含.*$", r"^涉及.*$", r"^提及.*$",
    r"^描述.*$", r"^讲清楚.*$", r"^确保.*$", r"^说明.*$", r"^指出.*$",
    r".*与医生沟通.*", r".*参考医生建议.*", r".*个体差异.*", r".*风险与收益.*",
]

def looks_generic(s: str) -> bool:
    s0 = s.strip()
    if len(s0) < 6:
        return True
    #  no medical entity/concept keywords and it is a template sentence
    lower = s0.lower()
    has_med_hint = any(k in lower for k in ["mg", "ml", "次/日", "周", "孕", "肝", "肾", "血压", "血糖", "禁忌", "不良反应", "药物", "症状", "诊断", "治疗", "禁用", "剂量", "并发症", "鉴别", "随访", "复查"])
    if not has_med_hint:
        for pat in GENERIC_PATTERNS:
            if re.match(pat, s0):
                return True
    return False

def normalize_for_dedup(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[，。、“”‘’!！?？：:;；,.()\[\]【】\-—\s]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

# --------------------------
# LLM plumbing
# --------------------------

SYSTEM_PROMPT = (
    "You are a careful medical evaluator. Review Chinese checklist items (rubrics) for a medical conversation.\n"
    "Rules:\n"
    "- DO NOT over-prune: if an item is relevant and not duplicative, KEEP it.\n"
    "- Remove items that are unrelated to the conversation/problem.\n"
    "- Merge semantically similar but complementary items into ONE, make it concise and more specific.\n"
    "- Remove overly generic/boilerplate items (e.g., '建议咨询医生', '注意个体差异', '确保回答清晰') unless grounded in the conversation.\n"
    "- If allowed, ADD up to K missing-but-important items strongly supported by the evidence passages.\n"
    "- Prefer to select an evidence_id from the provided ids for each kept/added item; if unsure, use 'EV_0'.\n"
)

USER_PROMPT_TEMPLATE = (
    "=== CONVERSATION ===\n"
    "{conversation_block}\n\n"
    "=== EVIDENCE (id -> content) ===\n"
    "{evidence_block}\n\n"
    "=== CANDIDATE RUBRICS (JSON array of strings) ===\n"
    "{rubrics_block}\n\n"
    "### Instructions\n"
    "1) Delete entries that are irrelevant to the conversation and retain those that are relevant and non-repetitive. Don't delete just to make things neat.\n"
    "2) Merge entries with similar but complementary semantics (retaining only one more specific Chinese expression).\n"
    "3) Delete overly broad and templated entries.\n"
    "4) If supplementation is allowed (K>0), up to K missing but critical entries can be supplemented based on evidence (avoid general statements and must be specific and verifiable).\n"
    "5) Specify evidence_id for each retained/merged/added entry (select from the list of ids above; use 'EV_0' if not sure).\n\n"
    "Return **strict JSON**（No additional explanations）：\n"
    "{{\n"
    "  \"kept\": [{{\"criterion\": \"...\", \"evidence_id\": \"p1\"}}, ...],\n"
    "  \"removed\": [{{\"criterion\": \"...\", \"reason\": \"...\"}}],\n"
    "  \"merged\": [{{\"from\": [\"c1\", \"c2\"], \"to\": {{\"criterion\": \"...\", \"evidence_id\": \"p1\"}}}}],\n"
    "  \"added\": [{{\"criterion\": \"...\", \"evidence_id\": \"p2\", \"reason\": \"strongly supported by [pid]\"}}]\n"
    "}}\n"
    "parameter：K={k_add}\n"
)

def load_model_and_tokenizer(model_name: str, dtype: str = "bfloat16"):
    torch_dtype = torch.bfloat16 if (dtype=="bfloat16" and torch.cuda.is_available()) else torch.float16
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto")
    pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device_map="auto")
    return tok, mdl, pipe

def robust_parse_json(text: str) -> Dict[str, Any]:
    # fenced
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # widest braces
    s = text.find("{"); e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        frag = text[s:e+1]
        try:
            return json.loads(frag)
        except Exception:
            return {}
    return {}

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_jsonl", type=str, required=True)
    ap.add_argument("--draft_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--summary_jsonl", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_new_tokens", type=int, default=900)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--allow_additions", action="store_true")
    ap.add_argument("--max_additions", type=int, default=2)
    ap.add_argument("--keep_top_n", type=int, default=0, help="optional cap after cleaning (0 = no cap)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.summary_jsonl) or ".")

    rag_rows   = read_jsonl(args.rag_jsonl)
    draft_rows = read_jsonl(args.draft_jsonl)

    # index -> conversation / evidence
    conv_map: Dict[int, str] = {}
    evid_map: Dict[int, Tuple[str, List[str]]] = {}
    for i, r in enumerate(rag_rows):
        idx = int(r.get("index", i))
        conv_map[idx] = build_conversation_text_from_prompt(r.get("prompt", []))
        evid_block, evid_ids = build_evidence_block(r.get("passages", []))
        evid_map[idx] = (evid_block, evid_ids)

    # index -> draft rubrics
    draft_map: Dict[int, List[Dict[str, Any]]] = {}
    for i, r in enumerate(draft_rows):
        idx = int(r.get("index", i))
        rubs = r.get("rubrics", [])
        if isinstance(rubs, list):
            draft_map[idx] = rubs

    tok, mdl, pipe = load_model_and_tokenizer(args.model_name)

    with open(args.summary_jsonl, "w", encoding="utf-8") as sum_f:
        for idx in sorted(set(conv_map.keys()) & set(draft_map.keys())):
            if idx < args.start or idx >= args.end:
                continue

            conversation_block = conv_map.get(idx, "")
            evidence_block, valid_ids = evid_map.get(idx, ("[EV_0] No external evidence provided.\n", ["EV_0"]))

            # Pass only the candidate rubrics (strings) to the LLM
            cands = [str(x.get("criterion","")).strip() for x in draft_map[idx] if str(x.get("criterion","")).strip()]
            # First, locally eliminate extremely broad items to relieve the burden on LLMS
            cands = [c for c in cands if not looks_generic(c)]

            rubrics_block = json.dumps(cands, ensure_ascii=False, indent=2)
            user_prompt = USER_PROMPT_TEMPLATE.format(
                conversation_block=conversation_block,
                evidence_block=evidence_block,
                rubrics_block=rubrics_block,
                k_add=(args.max_additions if args.allow_additions else 0),
            )
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}]
            if hasattr(tok, "apply_chat_template"):
                prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = str(messages)

            out = pipe(
                prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=(args.temperature > 0.0),
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
                return_full_text=False,
            )
            raw = out[0]["generated_text"] if out else ""
            obj = robust_parse_json(raw) or {}

            kept   = obj.get("kept",   [])
            merged = obj.get("merged", [])
            removed= obj.get("removed",[])
            added  = obj.get("added",  [])

            # normalize kept/added/merged.to  list[dict{criterion,evidence_id}]
            def _norm_item(it):
                if isinstance(it, dict):
                    c = str(it.get("criterion","")).strip()
                    e = str(it.get("evidence_id","EV_0")).strip() or "EV_0"
                    if valid_ids and e not in valid_ids:
                        e = "EV_0"
                    return {"criterion": c, "evidence_id": e} if c else None
                return None

            kept_norm = []
            for it in kept:
                if isinstance(it, dict):
                    kk = _norm_item(it)
                elif isinstance(it, str):
                    kk = _norm_item({"criterion": it, "evidence_id": "EV_0"})
                else:
                    kk = None
                if kk and not looks_generic(kk["criterion"]):
                    kept_norm.append(kk)

            merged_norm = []
            for m in merged:
                if not isinstance(m, dict): 
                    continue
                to = m.get("to")
                nn = _norm_item(to) if to else None
                if nn and not looks_generic(nn["criterion"]):
                    merged_norm.append({"to": nn, "from": m.get("from", [])})

            added_norm = []
            for it in added:
                kk = _norm_item(it) if isinstance(it, dict) else None
                if kk and not looks_generic(kk["criterion"]):
                    added_norm.append(kk)

            # merge kept + merged.to + added
            final_items = kept_norm + [m["to"] for m in merged_norm] + added_norm

            # Endgame string deduplication (guaranteed)
            seen, dedup = set(), []
            for it in final_items:
                k = normalize_for_dedup(it["criterion"])
                if not k or k in seen:
                    continue
                seen.add(k)
                dedup.append(it)

            # Optional limit quantity
            if args.keep_top_n > 0:
                dedup = dedup[:args.keep_top_n]

            result = {
                "index": idx,
                "kept": kept_norm,
                "merged": merged_norm,
                "added": added_norm,
                "removed": removed,
                "final": dedup,
                "raw_model_output": raw[:2000],
            }

            #  per-index
            out_path = os.path.join(args.out_dir, f"{idx}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            #  summary
            sum_f.write(json.dumps({"index": idx, "rubrics": dedup}, ensure_ascii=False) + "\n")

            print(f"[OK] index={idx} -> {out_path}  (final={len(dedup)}, kept={len(kept_norm)}, merged_to={len(merged_norm)}, added={len(added_norm)})")

if __name__ == "__main__":
    main()
