#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, sys, re, glob
from collections import Counter
from typing import List, Dict, Any, Tuple

# ------------------------------ Defaults ------------------------------
DEFAULT_SEMANTIC_TH = 0.80
DEFAULT_JACCARD_TH  = 0.60
MAX_DELETIONS_PER_INDEX = 3
ID_PAD = 6

AXES = {
    "accuracy", "completeness", "context_awareness",
    "communication_quality", "instruction_following"
}
AXIS_ALIASES = {
    "context": "context_awareness",
    "communication": "communication_quality",
    "instruction": "instruction_following",
    "instruction_follow": "instruction_following",
    "instr_follow": "instruction_following",
    "context-awareness": "context_awareness",
    "communication-quality": "communication_quality",
    "instr-follow": "instruction_following",
}

POLARITY_WORDS = set("""
correctly correct appropriate appropriately good well excellent
incorrect incorrectly wrong error errors failed fail fails failure
not no doesn't dont didn't didnot shouldn't shouldnt shouldnot should
avoid avoids avoided never none
miss missed misses missing
""".split())

BOILERPLATE_PHRASES = [
    "according to guidelines", "per guidelines",
    "as recommended", "as per recommendations",
    "in general", "in summary"
]

STOPWORDS = set("""
a an and are as at be but by for from has have if in into is it its
of on or that the their there these this to was were will with without
""".split())

# ------------------------------ Similarity backend ------------------------------
class SimilarityBackend:
    def __init__(self):

            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.mode = "st"


    def embed(self, texts: List[str]):
        if self.mode == "st":
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return texts

    def cos_sim(self, a, b) -> float:
        if self.mode == "st":
            import numpy as np
            return float(np.dot(a, b))
        return self.fuzz.token_set_ratio(a, b) / 100.0

# ------------------------------ Text utils ------------------------------
WS_RE = re.compile(r"\s+")
COMP_RE = re.compile(r"(?:>=|<=|>|<|\b\d+(?:\.\d+)?\b)")
TOKEN_RE = re.compile(r"[a-z0-9]+")

def norm_axis(a: str) -> str:
    if not a:
        return "accuracy"
    a = a.strip().lower()
    a = AXIS_ALIASES.get(a, a)
    return a if a in AXES else "accuracy"

def clean_ws(s: str) -> str:
    return WS_RE.sub(" ", s.strip())

def neutralize_polarity(s: str) -> str:
    s = s.lower()
    for ph in BOILERPLATE_PHRASES:
        s = s.replace(ph, " ")
    toks = [t for t in TOKEN_RE.findall(s) if t not in POLARITY_WORDS]
    return " ".join(toks)

def tokenize_for_jaccard(s: str) -> List[str]:
    return TOKEN_RE.findall(s.lower())

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

def has_numbers_or_conditions(s: str):
    hits = COMP_RE.findall(s)
    return (len(hits) > 0, len(hits))

def unique_content_words(a: str, b: str):
    A = set(w for w in tokenize_for_jaccard(a) if w not in STOPWORDS)
    B = set(w for w in tokenize_for_jaccard(b) if w not in STOPWORDS)
    return len(A - B), len(B - A)

def is_more_atomic(s: str):
    connectors = s.lower().count(" and ") + s.lower().count(" or ")
    return (connectors == 0, connectors)

def boilerplate_count(s: str) -> int:
    s_low = s.lower()
    return sum(int(ph in s_low) for ph in BOILERPLATE_PHRASES)

# ------------------------------ IO utils ------------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_inputs(path: str) -> List[Dict[str, Any]]:
    data = []
    if os.path.isdir(path):
        for fp in sorted(glob.glob(os.path.join(path, "*.json"))):
            with open(fp, "r", encoding="utf-8") as f:
                data.append(json.load(f))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data

def write_json(fp: str, obj: Any):
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def rewrite_jsonl_replace_indices(fp: str, new_records: List[Dict[str,Any]], key: str = "index"):
    """Replace existing lines (same `key`) in a JSONL, then append new lines."""
    keep_lines = []
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    try:
                        keep_lines.append(json.loads(s))
                    except Exception:
                        keep_lines.append(s)
    new_keys = {r.get(key) for r in new_records}
    keep_lines = [o for o in keep_lines if (isinstance(o, str) or o.get(key) not in new_keys)]
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for o in keep_lines:
            if isinstance(o, str):
                f.write(o + "\n")
            else:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, fp)

# ------------------------------ Core ------------------------------
def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    crit = clean_ws(str(item.get("criterion", "")))
    return {
        "criterion": crit,
        "axis": norm_axis(item.get("axis", "accuracy")),
        "point": int(item.get("point", 0)),
        "evidence_id": str(item.get("evidence_id")) if item.get("evidence_id") not in (None, "") else "",
        "criterion_neutral": neutralize_polarity(crit),
    }

def pairwise_candidates(norm_items: List[Dict[str, Any]]) -> List[Tuple[int,int]]:
    cand = []
    for i in range(len(norm_items)):
        ei, ai = norm_items[i]["evidence_id"], norm_items[i]["axis"]
        if not ei:
            continue
        for j in range(i+1, len(norm_items)):
            ej, aj = norm_items[j]["evidence_id"], norm_items[j]["axis"]
            if not ej:
                continue
            if ai != aj:
                continue
            if ei == ej:
                cand.append((i, j))
    return cand

def similarity_pass(a: Dict[str,Any], b: Dict[str,Any], backend, sem_th: float, jac_th: float):
    if backend.mode == "st":
        embs = backend.embed([a["criterion_neutral"], b["criterion_neutral"]])
        sim = backend.cos_sim(embs[0], embs[1])
    else:
        sim = backend.cos_sim(a["criterion_neutral"], b["criterion_neutral"])
    jac = jaccard(tokenize_for_jaccard(a["criterion_neutral"]), tokenize_for_jaccard(b["criterion_neutral"]))
    ok = (sim >= sem_th) and (jac >= jac_th)
    return ok, sim, jac

def choose_keep(a: Dict[str,Any], b: Dict[str,Any]) -> Tuple[str, str]:
    a_hasnum, a_n = has_numbers_or_conditions(a["criterion"])
    b_hasnum, b_n = has_numbers_or_conditions(b["criterion"])
    if a_hasnum != b_hasnum:
        return ("keep_a", "kept_more_specific_numbers") if a_hasnum else ("keep_b", "kept_more_specific_numbers")
    if a_hasnum and b_hasnum and a_n != b_n:
        return ("keep_a", "kept_more_specific_numbers") if a_n > b_n else ("keep_b", "kept_more_specific_numbers")
    au, bu = unique_content_words(a["criterion"], b["criterion"])
    if au != bu:
        return ("keep_a", "kept_more_unique_content_words") if au > bu else ("keep_b", "kept_more_unique_content_words")
    a_atomic, a_conn = is_more_atomic(a["criterion"]); b_atomic, b_conn = is_more_atomic(b["criterion"])
    if a_atomic != b_atomic:
        return ("keep_a", "kept_more_atomic") if a_atomic else ("keep_b", "kept_more_atomic")
    if a_conn != b_conn:
        return ("keep_a", "kept_more_atomic") if a_conn < b_conn else ("keep_b", "kept_more_atomic")
    ab, bb = boilerplate_count(a["criterion"]), boilerplate_count(b["criterion"])
    if ab != bb:
        return ("keep_a", "kept_less_boilerplate") if ab < bb else ("keep_b", "kept_less_boilerplate")
    if len(a["criterion"]) != len(b["criterion"]):
        return ("keep_a", "length_tiebreak") if len(a["criterion"]) > len(b["criterion"]) else ("keep_b", "length_tiebreak")
    return ("keep_a", "stable_order_fallback")

def process_one(sample: Dict[str,Any], backend, sem_th: float, jac_th: float, max_del: int,
                pairs_dir: str = None) -> Tuple[Dict[str,Any], Dict[str, int]]:
    index = sample.get("index")
    items_in = sample.get("rubrics") or sample.get("drafts") or []

    norm_items = [normalize_item(it) for it in items_in]
    original = len(norm_items)

    cands = pairwise_candidates(norm_items)
    removed = set()
    pairs_log = []

    # Score candidate pairs
    scored = []
    for (i, j) in cands:
        if not norm_items[i]["evidence_id"] or not norm_items[j]["evidence_id"]:
            continue
        ok, sim, jac = similarity_pass(norm_items[i], norm_items[j], backend, sem_th, jac_th)
        if ok:
            scored.append((sim + jac, i, j, sim, jac))
    scored.sort(reverse=True)

    # Preserve at least one negative if existed
    has_neg_before = any(int(ni["point"]) < 0 for ni in norm_items)

    for _, i, j, simv, jacv in scored:
        if i in removed or j in removed or len(removed) >= max_del:
            continue
        keep, reason = choose_keep(norm_items[i], norm_items[j])
        keep_idx, drop_idx = (i, j) if keep == "keep_a" else (j, i)
        if reason == "stable_order_fallback":
            keep_idx, drop_idx = (i, j) if i < j else (j, i)

        if has_neg_before:
            neg_after_if_drop = any(int(norm_items[k]["point"]) < 0
                                    for k in range(len(norm_items)) if (k not in removed and k != drop_idx))
            if not neg_after_if_drop:
                keep_idx, drop_idx = drop_idx, keep_idx
                reason = "preserve_negative_min1"

        removed.add(drop_idx)
        pairs_log.append({
            "pair_id": len(pairs_log) + 1,
            "axis": norm_items[i]["axis"],
            "evidence_id": norm_items[i]["evidence_id"],
            "semantic": round(simv, 4),
            "jaccard": round(jacv, 4),
            "keep_side": "left" if keep_idx == i else "right",
            "drop_side": "right" if keep_idx == i else "left",
            "reason": reason,
            "left_item": {
                "id": i,
                "criterion": norm_items[i]["criterion"],
                "axis": norm_items[i]["axis"],
                "point": norm_items[i]["point"],
                "evidence_id": norm_items[i]["evidence_id"]
            },
            "right_item": {
                "id": j,
                "criterion": norm_items[j]["criterion"],
                "axis": norm_items[j]["axis"],
                "point": norm_items[j]["point"],
                "evidence_id": norm_items[j]["evidence_id"]
            }
        })

    survivors = [ni for k, ni in enumerate(norm_items) if k not in removed]
    remaining = len(survivors)
    removed_count = original - remaining

    # Final items: include evidence_id (as string, may be empty if none)
    final_items = [{
        "criterion": ni["criterion"],
        "axis": ni["axis"],
        "point": ni["point"],
        "evidence_id": ni["evidence_id"] if ni["evidence_id"] != "" else None
    } for ni in survivors]

    # Save pairs log (pretty JSON array)
    if pairs_dir is not None:
        pairs_fp = os.path.join(pairs_dir, f"index_{int(index):0{ID_PAD}d}.pairs.json")
        with open(pairs_fp, "w", encoding="utf-8") as f:
            json.dump(pairs_log, f, ensure_ascii=False, indent=2)

    clean_obj = {
        "index": index,
        "final_rubrics": final_items
    }
    stats_obj = {
        "index": index,
        "original": original,
        "removed": removed_count,
        "remaining": remaining
    }
    return clean_obj, stats_obj

# ------------------------------ Runner ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Step3 conservative dedupe (final rubrics include evidence_id)")
    ap.add_argument("--input", required=True, help="Path to step2 JSONL or a directory of JSON files")
    ap.add_argument("--outdir", required=True, help="Output directory (no timestamp subfolder)")
    ap.add_argument("--start_index", type=int, default=None, help="Inclusive start index")
    ap.add_argument("--end_index", type=int, default=None, help="Inclusive end index")
    ap.add_argument("--semantic_th", type=float, default=DEFAULT_SEMANTIC_TH)
    ap.add_argument("--jaccard_th", type=float, default=DEFAULT_JACCARD_TH)
    ap.add_argument("--max_del", type=int, default=MAX_DELETIONS_PER_INDEX)
    ap.add_argument("--no_pairs_log", action="store_true", help="Do not save dedup pairs files")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    clean_dir = os.path.join(args.outdir, "clean_per_index")
    pairs_dir = None if args.no_pairs_log else os.path.join(args.outdir, "dedup_pairs_per_index")
    ensure_dir(clean_dir)
    if pairs_dir: ensure_dir(pairs_dir)

    final_jsonl_fp = os.path.join(args.outdir, "step3_final_rubrics.jsonl")
    counts_jsonl_fp = os.path.join(args.outdir, "step3_counts.jsonl")

    samples = read_inputs(args.input)
    if not samples:
        print("No input samples found.", file=sys.stderr)
        sys.exit(1)

    if args.start_index is not None and args.end_index is None:
        args.end_index = args.start_index
    if args.start_index is not None and args.end_index is not None and args.start_index > args.end_index:
        args.start_index, args.end_index = args.end_index, args.start_index

    def in_range(idx: int) -> bool:
        if args.start_index is None and args.end_index is None:
            return True
        return (idx is not None) and (args.start_index <= int(idx) <= args.end_index)

    samples = [s for s in samples if in_range(s.get("index"))]
    samples.sort(key=lambda x: x.get("index", 0))

    backend = SimilarityBackend()

    new_lines_rubrics = []
    new_lines_counts = []
    for sample in samples:
        clean_obj, stats_obj = process_one(
            sample, backend, args.semantic_th, args.jaccard_th, args.max_del,
            pairs_dir=pairs_dir
        )
        idx = int(clean_obj["index"])
        with open(os.path.join(clean_dir, f"index_{idx:0{ID_PAD}d}.rubrics.json"), "w", encoding="utf-8") as f:
            json.dump(clean_obj, f, ensure_ascii=False, indent=2)
        new_lines_rubrics.append({
            "index": clean_obj["index"],
            "rubrics": clean_obj["final_rubrics"]
        })
        new_lines_counts.append(stats_obj)

    if new_lines_rubrics:
        rewrite_jsonl_replace_indices(final_jsonl_fp, new_lines_rubrics, key="index")
    if new_lines_counts:
        rewrite_jsonl_replace_indices(counts_jsonl_fp, new_lines_counts, key="index")

    print(f"Done. Output directory: {args.outdir}")

if __name__ == "__main__":
    main()
