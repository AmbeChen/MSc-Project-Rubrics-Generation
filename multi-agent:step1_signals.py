# === multi_steps/step1_signals.py (part 1/5) ===
import os
import re
import json
import argparse
from typing import Any, Dict, List

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# Parse the model output JSON (fault-tolerant: with <json> wrapping, code blocks, trailing commas, etc.)
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].startswith("```"): lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s

def robust_json_extract(text: str) -> Dict[str, Any]:
    t = _strip_code_fences(text)
    if "<json>" in t and "</json>" in t:
        body = t.split("<json>", 1)[1].split("</json>", 1)[0]
        body = _strip_code_fences(body)
        body = re.sub(r",\s*([}\]])", r"\1", body)  # Remove the trailing comma
        return json.loads(body)
    # Fallback: Grab the outermost braces
    s = t.find("{"); e = t.rfind("}")
    if s != -1 and e != -1 and e > s:
        frag = _strip_code_fences(t[s:e+1])
        frag = re.sub(r",\s*([}\]])", r"\1", frag)
        return json.loads(frag)
    raise ValueError("Failed to parse JSON from model output")



# === (part 2/5) ===
def build_conversation(sample: Dict[str, Any]) -> str:
    # your data is usually [{"role":"user"/"assistant","content": "..."}]
    parts = []
    for t in sample.get("prompt", []):
        role = (t.get("role") or "user").upper()
        text = (t.get("content") or "").strip()
        parts.append(f"{role}: {text}")
    return "\n".join(parts)

SYSTEM_MSG = (
    "You are an assistant that prepares high-quality search queries for retrieval to help generate evaluation rubrics.\n"
    "Your job is NOT to write rubrics now, only to create queries for retrieval."
    "The retrieved information will be used as reference to help generate evaluation rubrics."
)

USER_PROMPT = """You will read a medical-style conversation and produce SEARCH QUERIES ONLY for retrieval.
Goal: help a later system generate evaluation rubrics (criteria) for this conversation, but this conversation is not complete lacking the final response, so you need to infer what the missing parts might be and provide help for clues for the whole conversation.

Return JSON with:
- "queries": a list of {n_queries} concise, high-quality search keywords/phrases/sentences.
   * They should be rubrics-oriented: reveal facts/rules/checklists that a rubric would verify.
   * The queries will be used to retrieve information from Mayo Clinic Site (a trusted medical information site covering Diseases & Conditions, Drugs & Supplements, Symptoms, Tests & Procedures, Healthy Lifestyle and Mayo Clinic Health Letter & Book). Write queries in a way that fits this source.
   * Use precise medical-relevant terms (drug names, conditions, dose/interval etc.) when clear from the conversation.
   * Suppose Only queries here will be used for retrieval, so please be careful and well define them, ensure the queries contain all necessary details and context.
   * Each query should be distinct and focused on a specific aspect of the conversation.
   * Do NOT include any site names (e.g., ‘Mayo Clinic’) in the queries.
- "entities": OPTIONAL; include ONLY when you are highly confident.
    * This must contain ONLY clinical entities
   * {{"drugs": [...], "diseases": [...], "conditions": ["..."], "procedures": [...]}} (omit a key if none).
Conversation:
===
{conversation}
===

Return only:
<json>
{{
  "queries": ["...", "..."],
  "entities": {{"drugs": ["..."], "diseases": [...], "conditions": ["..."]}}
}}
</json>
"""


# === (part 3/5) ===
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def build_generator(model_name: str, hf_token: str):
    tok = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token, device_map="auto", torch_dtype="auto", low_cpu_mem_usage=True
    )
    gen = pipeline("text-generation", model=mdl, tokenizer=tok)
    return gen, tok

def apply_chat(tok, messages):
    # Adapt to chat template (supports common instruction models like Llama3)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



# === (part 4/5) ===
def produce_signals_for_sample(
    sample: Dict[str, Any],
    gen,
    tok,
    n_queries: int = 4,
    max_new_tokens: int = 600
) -> Dict[str, Any]:
    conv = build_conversation(sample)
    prompt = USER_PROMPT.format(conversation=conv, n_queries=n_queries)
    messages = [{"role": "system", "content": SYSTEM_MSG},
                {"role": "user",   "content": prompt}]
    text = gen(apply_chat(tok, messages),
               max_new_tokens=max_new_tokens,
               temperature=0.0,
               do_sample=False,
               return_full_text=False)[0]["generated_text"]
    try:
        obj = robust_json_extract(text)
    except Exception:
        obj = {}

    # Guarantee: Ensure that queries exist and the quantity is compliant
    queries = obj.get("queries") or []
    if not isinstance(queries, list):
        queries = []
    # Simple fallback: Extract a few keywords from the last user message to form 2-3 queries
    if not queries:
        last_user = ""
        for t in reversed(sample.get("prompt", [])):
            if (t.get("role") or "user") == "user":
                last_user = t.get("content") or ""
                break
        tokens = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", last_user)]
        # Combine a few groups of keywords (very simple but effective)
        fallback = []
        if tokens:
            fallback.append(" ".join(tokens[:6]))
            if len(tokens) > 6:
                fallback.append(" ".join(tokens[6:12]))
        queries = fallback[:max(2, n_queries//2)]

    # Truncate overly long queries and remove duplicates
    norm = []
    seen = set()
    for q in queries:
        q = (q or "").strip()
        if not q: continue
        q = re.sub(r"\s+", " ", q)
        if len(q.split()) > 16:
            q = " ".join(q.split()[:16])
        if q.lower() in seen: 
            continue
        seen.add(q.lower())
        norm.append(q)
    # Control quantity (strictly n_queries)
    if len(norm) > n_queries:
        norm = norm[:n_queries]
    elif len(norm) < n_queries and queries:
        # Repeat to fill
        while len(norm) < n_queries:
            norm.append(queries[min(len(norm), len(queries)-1)])

    entities = obj.get("entities") if isinstance(obj.get("entities"), dict) else {}

    return {
        "queries": norm,
        "entities": entities
    }



# === (part 5/5) ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="multi_steps/data/rubrics_8_10_short.jsonl")
    ap.add_argument("--index", type=int, default=0, help="index for single mode")
    ap.add_argument("--batch", action="store_true", help="if set, run for all rows")
    ap.add_argument("--out_path", type=str, default="multi_steps_v2/outputs/step1_signals.json",
                    help="single-output path")
    ap.add_argument("--out_all_path", type=str, default="multi_steps_v2/outputs/step1_signals_all.jsonl",
                    help="batch-output path (JSONL)")
    ap.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--n_queries", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=600)
    args = ap.parse_args()

    rows = load_jsonl(args.data_path)
    gen, tok = build_generator(args.model_name, args.hf_token)

    if args.batch:
        # Full amount: Write one line for each dialogue
        if os.path.exists(args.out_all_path):
            os.remove(args.out_all_path)
        for i, row in enumerate(rows):
            sig = produce_signals_for_sample(row, gen, tok,
                                             n_queries=args.n_queries,
                                             max_new_tokens=args.max_new_tokens)
            sig["index"] = i
            append_jsonl(sig, args.out_all_path)
        print(f"[OK] wrote batch signals -> {args.out_all_path} ({len(rows)} lines)")
    else:
        # Single entry: Write a JSON
        if args.index < 0 or args.index >= len(rows):
            raise IndexError(f"index out of range: 0..{len(rows)-1}")
        sig = produce_signals_for_sample(rows[args.index], gen, tok,
                                         n_queries=args.n_queries,
                                         max_new_tokens=args.max_new_tokens)
        sig["index"] = args.index
        save_json(sig, args.out_path)
        print(f"[OK] wrote single signals -> {args.out_path}")

if __name__ == "__main__":
    main()
