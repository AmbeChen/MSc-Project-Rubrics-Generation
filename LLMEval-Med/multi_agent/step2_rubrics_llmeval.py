#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2 (LLMEval-Med): Generate draft rubrics (checklists) from conversations + RAG JSONL + few-shot.

Differences from HealthBench:
- Conversation is read directly from `prompt` field in JSONL (role/content pairs).
- Rubrics format simplified: only 'criterion' (string) + 'evidence_id' (string).
- Output JSON, Markdown, and summary JSONL are still supported.
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
# Conversation (from 'prompt')
# --------------------------

def read_conversation_file(conversation_dir: str, template: str, index: int) -> str:
    path = os.path.join(conversation_dir, template.format(index=index))
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# --------------------------
# Evidence (from 'passages')
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
        return "[EV_0] No external evidence provided.\n", ["EV_0"]

    blocks = []
    ids: List[str] = []
    for i, doc in enumerate(passages):
        evid = str(doc.get("id", f"EV_{i}"))
        title = str(doc.get("title", doc.get("source", "")))
        body = str(doc.get("text", doc.get("snippet", "")))
        if len(body) > max_chars_per_doc:
            body = body[:max_chars_per_doc] + " ..."
        url = str(doc.get("url", ""))
        block = f"[{evid}] {title}\n{body}\n{url}".strip()
        blocks.append(block)
        ids.append(evid)

    return "\n\n".join(blocks) + "\n", ids

# --------------------------
# Few-shot (LLMEval-Med style)
# --------------------------

def build_fewshot_block(fewshots: List[Dict[str, Any]], k: int = 2) -> str:
    exemplars = []
    used = 0
    for fs in fewshots:
        if used >= k:
            break
        prob = fs.get("problem", "")
        rubs = fs.get("checklist", [])
        rub_texts = [f"- {r}" for r in rubs if isinstance(r, str) and r.strip()]
        exemplars.append(
            f"### Example Problem\n{prob}\n"
            f"### Example Rubrics\n" + "\n".join(rub_texts)
        )
        used += 1
    return "\n\n".join(exemplars) if exemplars else "(No exemplars)"

# --------------------------
# Prompting
# --------------------------

SYSTEM_PROMPT = (
        "You are a medical evaluation assistant. Your task is to generate rubrics (evaluation criteria) for assessing model responses in medical conversations.\n"
        "You will be given EXAMPLES of how to generate rubrics. Then, you will be asked to generate rubrics for a NEW conversation.\n\n"
        "Each rubric must:\n"
        "- Be a clear, actionable evaluation criterion.\n"
        "- Directly relate to the given conversation and its medical content.\n"
        " Rubrics should be checklist items, used to verify whether the responses cover key medical information.\n"
        " Each rubric item must be a specific knowledge point or checklist item related to the medical content. Rather than the general accuracy/completeness/communication items.\n"
        " If you find the provided reference materials helpful, please align the key points and expressions therein. If you determine that the reference information is irrelevant, please ignore it. Generate the rubrics/checklist items on your own and based on your medical knowledge.\n"
)

USER_PROMPT_TEMPLATE = (
    "=== FEW-SHOT EXAMPLES ===\n"
    "{fewshot_block}\n\n"
    "=== TARGET CONVERSATION ===\n"
    "{conversation_block}\n\n"
    "=== Reference Information / Evidence (id -> content) ===\n"
    "{evidence_block}\n\n"
    "### Task\n"
    "Please generate exactly **10 rubrics in CHINESE**. Each rubric must be:\n"
    "- A clear, specific checklist item related to the conversation.\n"
    "- Directly tied to the medical content.\n"
    "- Each rubric must be **directly related to the specific conversation content**. Do not include generic or unrelated criteria.\n"
    "- If supported by evidence, assign the corresponding evidence_id; otherwise use 'EV_0'.\n\n"
    "Return JSON array like:\n"
    "[{{\"criterion\": \"...\", \"evidence_id\": \"p1\"}}, ...]\n"
)

# --------------------------
# Parse & normalize
# --------------------------

def parse_json_safely(text: str):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\[.*\])", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return []
    return []

def normalize_rubrics(items: Any, valid_evid_ids: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out = []
    fallback = valid_evid_ids[0] if valid_evid_ids else "EV_0"
    for it in items:
        if not isinstance(it, dict):
            continue
        criterion = str(it.get("criterion", "")).strip()
        evid = str(it.get("evidence_id", fallback)).strip() or fallback
        if evid not in valid_evid_ids:
            evid = fallback
        if criterion:
            out.append({"criterion": criterion, "evidence_id": evid})
    return out

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
# Human-readable draft (Markdown)
# --------------------------

def save_readable_markdown(path: str, conversation_block: str, evidence_block: str, rubrics: List[Dict[str, Any]], prompt_text: str) -> None:
    lines = []
    lines.append("# Rubrics Draft (LLMEval-Med)\n")
    lines.append("## Prompt Sent to Model\n")
    lines.append("```\n" + prompt_text[:4000] + "\n```\n")  # limit length to avoid huge files
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
            lines.append(f"- **criterion:** {r.get('criterion','')}")
            lines.append(f"- **evidence_id:** {r.get('evidence_id','')}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_jsonl", type=str, required=True)
    ap.add_argument("--fewshot_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--summary_jsonl", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--fewshot_k", type=int, default=2)
    ap.add_argument("--conversation_dir", type=str, default="outputs/prompts")
    ap.add_argument("--conversation_template", type=str, default="conversation_{index}.txt")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--max_new_tokens", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.summary_jsonl) or ".")

    rag_data = read_jsonl(args.rag_jsonl)
    fewshot_data = read_jsonl(args.fewshot_jsonl)
    fewshot_block = build_fewshot_block(fewshot_data, k=args.fewshot_k)

    tokenizer, model, pipe = load_model_and_tokenizer(args.model_name)

    summary_map = {}

    with open(args.summary_jsonl, "a", encoding="utf-8") as sum_f:
        for i, ex in enumerate(rag_data):
            ex_idx = extract_index(ex, fallback=i)
            if ex_idx < args.start or ex_idx >= args.end:
                continue

            # conversation
            raw_conv = read_conversation_file(args.conversation_dir, args.conversation_template, ex_idx)
            if raw_conv:
                conversation_block = raw_conv
            else:
                conversation_block = "(No conversation provided)"

            # evidence
            passages = extract_passages(ex)
            evidence_block, valid_evid_ids = build_evidence_from_passages(passages)

            # prompt
            user_prompt = USER_PROMPT_TEMPLATE.format(
                fewshot_block=fewshot_block,
                evidence_block=evidence_block,
                conversation_block=conversation_block,
            )
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}]
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = str(messages)

            # generate
            gen = pipe(
                prompt_text,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0.0),
                temperature=args.temperature,
                return_full_text=False,
            )
            raw_text = gen[0]["generated_text"] if gen else ""

            parsed = parse_json_safely(raw_text)
            rubrics = normalize_rubrics(parsed, valid_evid_ids=valid_evid_ids)

            # save outputs
            per = {"index": ex_idx, "rubrics": rubrics, "raw_model_output": raw_text}
            json_path = os.path.join(args.out_dir, f"{ex_idx}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(per, f, ensure_ascii=False, indent=2)

            summary_map[ex_idx] = {"index": ex_idx, "rubrics": rubrics}
            sum_f.write(json.dumps({"index": ex_idx, "rubrics": rubrics}, ensure_ascii=False) + "\n")
            
            md_path = os.path.join(args.out_dir, f"{ex_idx}.md")
            save_readable_markdown(md_path, conversation_block, evidence_block, rubrics, prompt_text)


            print(f"[OK] index={ex_idx} -> {json_path}")

if __name__ == "__main__":
    main()
