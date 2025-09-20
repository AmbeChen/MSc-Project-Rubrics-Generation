# === file: multi_steps_v2/step1_signals_llmevalmed.py ===
import os
import re
import json
import argparse
from typing import Any, Dict, List

# ---------------------------
# IO helpers
# ---------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows

def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------------------------
# Robust JSON extraction
# ---------------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s

def robust_json_extract(text: str) -> Dict[str, Any]:
    t = _strip_code_fences(text)
    if "<json>" in t and "</json>" in t:
        body = t.split("<json>", 1)[1].split("</json>", 1)[0]
        body = _strip_code_fences(body)
        body = re.sub(r",\s*([}\]])", r"\1", body)  # remove trailing commas
        return json.loads(body)
    s = t.find("{"); e = t.rfind("}")
    if s != -1 and e != -1 and e > s:
        frag = _strip_code_fences(t[s:e+1])
        frag = re.sub(r",\s*([}\]])", r"\1", frag)
        return json.loads(frag)
    raise ValueError("Failed to parse JSON from model output")

# ---------------------------
# Prompt for LLMEval-Med (problem-only)
# ---------------------------
SYSTEM_MSG = (
    "You are an assistant that prepares high-quality Chinese search queries for medical retrieval.\n"
    "Your job is NOT to write rubrics now, only to create queries for retrieval.\n"
    "The retrieved information will later help generate evaluation rubrics."
)

USER_PROMPT = """You will see a ** medical question（problem）**，Please generate a ** Chinese search query ** only for the retrieval system（queries），Used for retrieving evidence from authoritative medical databases（Priority: diseases/symptoms/examinations/medications/healthy lifestyles, etc.) **Do not include any site names in the query**。

Goal：Help the subsequent system generate an evaluation list for this problem (rubrics). Please construct a search query around the facts, mechanisms, key points of identification, risk points, key points of treatment/medication, and key points of examination that may need to be verified regarding this issue.

Return JSON (strict)：
- "queries": totally {n_queries} **Chinese** search queries，each query focuses on one aspect. Try to include medical keywords (disease name/drug name/pathological mechanism/dosage or frequency/population/time window, etc.) to facilitate hitting authoritative health items.
- "entities":(Optional) Extract clinical entities if and only if you are very certain：
    {{
      "drugs": [...], "diseases": [...], "conditions": [...], "procedures": [...]
    }}（If not, omit the corresponding key.）

Problem:
===
{problem}
===

Only return：
<json>
{{
  "queries": ["...", "...", "...", "..."],
  "entities": {{"drugs": ["..."], "diseases": ["..."], "conditions": ["..."]}}
}}
</json>
"""

# ---------------------------
# HF / LLM
# ---------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def build_generator(model_name: str, hf_token: str):
    tok = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token, device_map="auto", torch_dtype="auto", low_cpu_mem_usage=True
    )
    gen = pipeline("text-generation", model=mdl, tokenizer=tok)
    return gen, tok

def apply_chat(tok, messages):

    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ---------------------------
# Core
# ---------------------------
def produce_signals_for_problem(
    problem_text: str,
    gen,
    tok,
    n_queries: int = 4,
    max_new_tokens: int = 500
) -> Dict[str, Any]:
    prompt = USER_PROMPT.format(problem=problem_text, n_queries=n_queries)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": prompt}
    ]
    out = gen(
        apply_chat(tok, messages),
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        return_full_text=False
    )[0]["generated_text"]

    # parse JSON
    try:
        obj = robust_json_extract(out)
    except Exception:
        obj = {}

    # Ensure the existence of queries
    queries = obj.get("queries") or []
    if not isinstance(queries, list):
        queries = []

    # Failure guarantee: Extract Chinese keywords from "problem" and combine them to create 2-3 queries
    if not queries:
        txt = problem_text
        # Try to extract noun phrases from Chinese text (simple fallback)
        # word segmentation
        toks = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", txt)
        toks = [t for t in toks if len(t) >= 2]
        fallback = []
        if toks:
            fallback.append(" ".join(toks[:6]))
            if len(toks) > 6:
                fallback.append(" ".join(toks[6:12]))
        queries = fallback[:max(2, n_queries//2)]

    # clean + dedupe + cut
    norm, seen = [], set()
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        q = re.sub(r"\s+", " ", q)

        if len(q) > 80:
            q = q[:80]
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        norm.append(q)

    # number control
    if len(norm) > n_queries:
        norm = norm[:n_queries]
    elif len(norm) < n_queries and norm:
        while len(norm) < n_queries:
            norm.append(norm[len(norm) % len(norm)])

    entities = obj.get("entities") if isinstance(obj.get("entities"), dict) else {}

    return {
        "queries": norm,
        "entities": entities
    }

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/problems_checklist_clean.jsonl",
                    help="JSONL with at least a 'problem' field per line")
    ap.add_argument("--index", type=int, default=0, help="index for single mode")
    ap.add_argument("--batch", action="store_true", help="run for all rows if set")
    ap.add_argument("--out_path", type=str, default="multi_agent/outputs/step1_signals.json",
                    help="single-output path")
    ap.add_argument("--out_all_path", type=str, default="multi_agent/outputs/step1_signals_all.jsonl",
                    help="batch-output path (JSONL)")
    ap.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--n_queries", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=500)
    args = ap.parse_args()

    rows = load_jsonl(args.data_path)
    gen, tok = build_generator(args.model_name, args.hf_token)

    def get_problem(row: Dict[str, Any]) -> str:
    
        if "problem" in row:
            return (row.get("problem") or "").strip()
        # fallback
        prompt_turns = row.get("prompt") or []
        if prompt_turns:
            lines = []
            for t in prompt_turns:
                role = (t.get("role") or "user").upper()
                text = (t.get("content") or "").strip()
                lines.append(f"{role}: {text}")
            return "\n".join(lines)
        return ""

    if args.batch:
        if os.path.exists(args.out_all_path):
            os.remove(args.out_all_path)
        for i, row in enumerate(rows):
            pb = get_problem(row)
            sig = produce_signals_for_problem(pb, gen, tok,
                                              n_queries=args.n_queries,
                                              max_new_tokens=args.max_new_tokens)
            sig["index"] = i
            sig["problem"] = pb
            append_jsonl(sig, args.out_all_path)
        print(f"[OK] wrote batch signals -> {args.out_all_path} ({len(rows)} lines)")
    else:
        if args.index < 0 or args.index >= len(rows):
            raise IndexError(f"index out of range: 0..{len(rows)-1}")
        pb = get_problem(rows[args.index])
        sig = produce_signals_for_problem(pb, gen, tok,
                                          n_queries=args.n_queries,
                                          max_new_tokens=args.max_new_tokens)
        sig["index"] = args.index
        sig["problem"] = pb
        save_json(sig, args.out_path)
        print(f"[OK] wrote single signals -> {args.out_path}")

if __name__ == "__main__":
    main()
