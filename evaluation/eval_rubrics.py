# -*- coding: utf-8 -*-
"""
Embedding-based Rubrics Evaluation (index-first reference lookup)
- Prioritize per-index ref files: REF_DIR/ref_rubrics_{index}.json
- Fallback to JSONL when per-index file is missing
- Embeddings via:
    1) SentenceTransformer(model_name) when possible
    2) Transformers AutoModel + mean pooling (fallback)
- Multi-to-multi matching: threshold + top-k pruning (optional mutual-topk)
- Axis-aware: strict (mask) or bonus (add)
- Outputs:
    * Per-index pairs under OUT_DIR/idx_{index}.json
    * Single overall metrics file (OVERALL_JSON/OVERALL_CSV)
    * Added p-values: t-test & Wilcoxon on per-index F1; Wilcoxon on (edge_sim - threshold)
"""

import os
import re
import json
import csv
import time
import argparse
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_1samp, wilcoxon

# -------------------- IO helpers --------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                print(f"[WARN] {path}:{ln} invalid json: {e}")
    return rows

def read_json_any(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] cannot read {path}: {e}")
        return None

# -------------------- Data normalization --------------------
AXIS_RE = re.compile(r"^axis:(.+)$", re.I)

def extract_axis(tags: Any) -> str:
    if isinstance(tags, list):
        for t in tags:
            if isinstance(t, str):
                m = AXIS_RE.match(t.strip())
                if m:
                    return m.group(1).strip().lower()
    if isinstance(tags, str):
        m = AXIS_RE.match(tags.strip())
        if m:
            return m.group(1).strip().lower()
    return "unknown"

def coerce_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(round(float(x)))
        except Exception:
            return default

def norm_item(d: Dict[str, Any]) -> Dict[str, Any]:
    crit = (d.get("criterion") or d.get("criteria") or d.get("text") or "").strip()
    pts = coerce_int(d.get("points", d.get("point", 0)))
    tags = d.get("tags") or []
    axis = (d.get("axis") or extract_axis(tags) or "unknown").strip().lower()
    return {"criterion": crit, "points": pts, "axis": axis, "tags": tags}

def extract_gen_row(row: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
    idx = row.get("index", row.get("conversation_id"))
    # common containers
    for k in ("rubrics", "generated_rubrics", "final_rubrics", "pred_rubrics", "items", "output", "data"):
        v = row.get(k)
        if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
            return idx, [norm_item(x) for x in v if isinstance(x, dict)]
    # nested
    if isinstance(row.get("rubrics"), dict):
        for kk in ("items", "list", "data"):
            v = row["rubrics"].get(kk)
            if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
                return idx, [norm_item(x) for x in v if isinstance(x, dict)]
    # fallback: first list-of-dicts with criterion-like fields
    for v in row.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and any(k in v[0] for k in ("criterion","criteria","text")):
            return idx, [norm_item(x) for x in v if isinstance(x, dict)]
    return idx, []

# -------------------- Reference lookup --------------------
def build_fallback_map(fallback_jsonl: str) -> Dict[Any, List[Dict[str, Any]]]:
    m: Dict[Any, List[Dict[str, Any]]] = {}
    rows = read_jsonl(fallback_jsonl)
    for i, item in enumerate(rows):
        key = item.get("index", item.get("conversation_id", i))
        rubs = item.get("rubrics") or item.get("items") or item.get("data") or []
        if isinstance(rubs, dict):
            rubs = [rubs]
        if isinstance(rubs, list):
            rubs = [norm_item(x) for x in rubs if isinstance(x, dict)]
        else:
            rubs = []
        if key not in m:
            m[key] = []
        m[key].extend(rubs)
    if os.path.exists(fallback_jsonl):
        print(f"[INFO] fallback map built: {len(m)} keys from {fallback_jsonl}")
    return m

def load_ref_for_index(idx: Any, ref_dir: str, fallback_map: Dict[Any, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    ref_path = os.path.join(ref_dir, f"ref_rubrics_{idx}.json")
    if os.path.exists(ref_path):
        data = read_json_any(ref_path)
        if isinstance(data, dict):
            data = data.get("rubrics") or data.get("items") or data.get("data") or []
        if isinstance(data, list):
            return [norm_item(x) for x in data if isinstance(x, dict)]
        return []
    return fallback_map.get(idx, [])

# -------------------- Embedding loader (dual-path, token-aware) --------------------
def load_encoder(model_name: str, device: str, hf_token: Optional[str], batch_size: int, normalize_embeddings: bool):
    """
    Returns a callable: encode(texts: List[str]) -> np.ndarray [n, d] (L2 normalized if normalize_embeddings=True)
    1) Try SentenceTransformer
    2) Fallback to Transformers AutoModel + mean pooling
    """
    # try sentence-transformers path
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[INFO] loading SentenceTransformer: {model_name} on {device}")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        st_model = SentenceTransformer(model_name, device=device)  # may raise if not sbert-format

        def encode_st(texts: List[str]) -> np.ndarray:
            if not texts:
                return np.zeros((0, st_model.get_sentence_embedding_dimension()), dtype=np.float32)
            embs = st_model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False
            )
            if not normalize_embeddings:
                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
                embs = embs / norms
            return embs.astype(np.float32)

        return encode_st

    except Exception as e:
        print(f"[INFO] SentenceTransformer path failed ({e}). Falling back to Transformers AutoModel.")

    # transformers AutoModel + mean pooling
    import torch
    from transformers import AutoTokenizer, AutoModel

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    print(f"[INFO] loading Transformers AutoModel: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
    model = AutoModel.from_pretrained(model_name, token=hf_token)
    model = model.to(device)
    model.eval()

    @torch.no_grad()
    def encode_tm(texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, model.config.hidden_size), dtype=np.float32)
        out_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            out = model(**enc)
            last = out.last_hidden_state  # [B, L, H]
            attn = enc["attention_mask"].unsqueeze(-1)  # [B, L, 1]
            summed = (last * attn).sum(dim=1)          # [B, H]
            lens = attn.sum(dim=1).clamp(min=1)        # [B, 1]
            vec = (summed / lens).cpu().numpy()        # mean-pooled
            norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
            vec = (vec / norms).astype(np.float32)
            out_list.append(vec)
        return np.vstack(out_list) if out_list else np.zeros((0, model.config.hidden_size), dtype=np.float32)

    return encode_tm

# -------------------- Similarity & matching --------------------
def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    return np.matmul(A, B.T).astype(np.float32)

def build_edges(sim: np.ndarray,
                gen_axes: List[str],
                ref_axes: List[str],
                threshold: float,
                axis_strict: bool,
                axis_bonus: float,
                topk_ref_per_gen: int,
                topk_gen_per_ref: int,
                mutual_topk: bool) -> List[Tuple[int,int,float,bool]]:
    """
    Multi-to-multi edges with threshold + top-k pruning.
    Returns: list of (i, j, score, axis_match)
    """
    G, R = sim.shape
    if G == 0 or R == 0:
        return []

    adj = sim.copy()
    # axis policy
    for i in range(G):
        for j in range(R):
            same_ax = (gen_axes[i] == ref_axes[j])
            if axis_strict and not same_ax:
                adj[i, j] = -1.0  # mask
            elif (not axis_strict) and same_ax:
                adj[i, j] = min(1.0, adj[i, j] + axis_bonus)

    # top-k per row (gen)
    keep_row = np.zeros_like(adj, dtype=bool)
    for i in range(G):
        row = adj[i]
        idxs = np.argsort(row)[::-1][:topk_ref_per_gen]
        keep_row[i, idxs] = True

    # top-k per col (ref)
    keep_col = np.zeros_like(adj, dtype=bool)
    for j in range(R):
        col = adj[:, j]
        idxs = np.argsort(col)[::-1][:topk_gen_per_ref]
        keep_col[idxs, j] = True

    # candidate mask
    cand = keep_row & keep_col
    if mutual_topk:
        pass
    else:
        pass  # keep intersection; switch to union if you want looser matching

    edges: List[Tuple[int,int,float,bool]] = []
    for i in range(G):
        for j in range(R):
            if not cand[i, j]:
                continue
            score = float(adj[i, j])
            if score >= threshold:
                edges.append((i, j, score, gen_axes[i] == ref_axes[j]))
    return edges

# -------------------- Metrics --------------------
def node_level_metrics(edges: List[Tuple[int,int,float,bool]], G: int, R: int) -> Dict[str, float]:
    if G == 0 or R == 0:
        return {"precision_node": 0.0, "recall_node": 0.0, "f1_node": 0.0}
    matched_gen = {i for (i, _, _, _) in edges}
    matched_ref = {j for (_, j, _, _) in edges}
    p = len(matched_gen) / G
    r = len(matched_ref) / R
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision_node": p, "recall_node": r, "f1_node": f1}

def best_similarity_scores(sim: np.ndarray) -> Tuple[float, float]:
    if sim.size == 0:
        return 0.0, 0.0
    best_ref = float(np.mean(np.max(sim, axis=0))) if sim.shape[0] > 0 else 0.0
    best_gen = float(np.mean(np.max(sim, axis=1))) if sim.shape[1] > 0 else 0.0
    return best_ref, best_gen

def mae_points(edges: List[Tuple[int,int,float,bool]], gen_pts: List[int], ref_pts: List[int]) -> float:
    if not edges:
        return 0.0
    diffs = [abs(gen_pts[i] - ref_pts[j]) for (i, j, _, _) in edges]
    return float(np.mean(diffs)) if diffs else 0.0

def axis_match_rate(edges: List[Tuple[int,int,float,bool]]) -> float:
    if not edges:
        return 0.0
    same = sum(1 for (_, _, _, s) in edges if s)
    return same / len(edges)

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("--gen_path", required=True, type=str)
    parser.add_argument("--ref_dir", required=True, type=str)
    parser.add_argument("--ref_fallback", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--overall_json", required=True, type=str)
    parser.add_argument("--overall_csv", required=True, type=str)
    # model & runtime
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", type=str, default="cuda")  # "cuda" or "cpu"
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--normalize_embeddings", type=int, choices=[0,1], default=1)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", None))
    # matching
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--topk_ref_per_gen", type=int, default=3)
    parser.add_argument("--topk_gen_per_ref", type=int, default=3)
    parser.add_argument("--mutual_topk", type=int, choices=[0,1], default=0)
    # axis policy
    parser.add_argument("--axis_strict", type=int, choices=[0,1], default=0)
    parser.add_argument("--axis_bonus", type=float, default=0.05)

    args = parser.parse_args()

    # prep dirs
    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.overall_json))
    ensure_dir(os.path.dirname(args.overall_csv))

    cfg = {
        "model": args.model,
        "device": args.device,
        "batch_size": args.batch_size,
        "normalize_embeddings": bool(args.normalize_embeddings),
        "threshold": args.threshold,
        "topk_ref_per_gen": args.topk_ref_per_gen,
        "topk_gen_per_ref": args.topk_gen_per_ref,
        "mutual_topk": bool(args.mutual_topk),
        "axis_strict": bool(args.axis_strict),
        "axis_bonus": args.axis_bonus,
    }
    print("[INFO] config:", json.dumps(cfg, indent=2))

    # load data
    gen_rows = read_jsonl(args.gen_path)
    if not gen_rows:
        raise FileNotFoundError(f"No rows found in {args.gen_path}")
    fb_map = build_fallback_map(args.ref_fallback)

    # load encoder (dual path)
    encode = load_encoder(
        model_name=args.model,
        device=args.device,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
        normalize_embeddings=bool(args.normalize_embeddings),
    )

    # global caches and aggregates
    text_cache: Dict[str, np.ndarray] = {}

    def get_vecs(texts: List[str]) -> np.ndarray:
        missing = [t for t in texts if t not in text_cache]
        if missing:
            embs = encode(missing)
            for t, e in zip(missing, embs):
                text_cache[t] = e
        return np.stack([text_cache[t] for t in texts], axis=0) if texts else np.zeros((0,1), dtype=np.float32)

    total_p_node: List[float] = []
    total_r_node: List[float] = []
    total_f1_node: List[float] = []
    total_best_ref: List[float] = []
    total_best_gen: List[float] = []
    total_edge_sims: List[float] = []
    total_axis_flags: List[bool] = []
    total_mae_list: List[float] = []

    processed = 0
    skipped = 0
    t0 = time.time()

    for row in tqdm(gen_rows, desc="Evaluating"):
        idx, gen_items = extract_gen_row(row)
        gen_items = [norm_item(x) for x in gen_items]
        ref_items = load_ref_for_index(idx, args.ref_dir, fb_map)

        if not gen_items or not ref_items:
            out_path = os.path.join(args.out_dir, f"idx_{idx}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"index": idx, "pairs": [], "uncovered_ref": [], "stats": {"note": "empty gen or ref"}}, f, ensure_ascii=False, indent=2)
            skipped += 1
            continue

        gen_texts = [g["criterion"] for g in gen_items]
        ref_texts = [r["criterion"] for r in ref_items]
        gen_axes = [g["axis"] for g in gen_items]
        ref_axes = [r["axis"] for r in ref_items]
        gen_pts = [g["points"] for g in gen_items]
        ref_pts = [r["points"] for r in ref_items]

        Vg = get_vecs(gen_texts)
        Vr = get_vecs(ref_texts)
        S = cosine_matrix(Vg, Vr)

        edges = build_edges(
            sim=S,
            gen_axes=gen_axes,
            ref_axes=ref_axes,
            threshold=args.threshold,
            axis_strict=bool(args.axis_strict),
            axis_bonus=args.axis_bonus,
            topk_ref_per_gen=args.topk_ref_per_gen,
            topk_gen_per_ref=args.topk_gen_per_ref,
            mutual_topk=bool(args.mutual_topk)
        )

        # per-index stats for file
        node_m = node_level_metrics(edges, len(gen_items), len(ref_items))
        best_ref, best_gen = best_similarity_scores(S)
        mae = mae_points(edges, gen_pts, ref_pts)
        amr = axis_match_rate(edges)

        pairs_json = [
            {
                "gen_idx": i,
                "ref_idx": j,
                "gen": gen_items[i],
                "ref": ref_items[j],
                "similarity": round(s, 4),
                "axis_match": same_ax
            }
            for (i, j, s, same_ax) in edges
        ]
        uncovered_ref = [ref_items[j] for j in range(len(ref_items)) if all(e[1] != j for e in edges)]

        out_path = os.path.join(args.out_dir, f"idx_{idx}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "index": idx,
                "pairs": pairs_json,
                "uncovered_ref": uncovered_ref,
                "stats": {
                    "G": len(gen_items), "R": len(ref_items),
                    "edges": len(edges),
                    **{k: round(v, 4) for k, v in node_m.items()},
                    "best_sim_ref": round(best_ref, 4),
                    "best_sim_gen": round(best_gen, 4),
                    "mae_points": round(mae, 4),
                    "axis_match_rate": round(amr, 4),
                    "threshold": args.threshold,
                    "topk_ref_per_gen": args.topk_ref_per_gen,
                    "topk_gen_per_ref": args.topk_gen_per_ref,
                    "axis_strict": bool(args.axis_strict),
                    "axis_bonus": args.axis_bonus,
                    "mutual_topk": bool(args.mutual_topk)
                }
            }, f, ensure_ascii=False, indent=2)

        # collect for overall
        total_p_node.append(node_m["precision_node"])
        total_r_node.append(node_m["recall_node"])
        total_f1_node.append(node_m["f1_node"])
        total_best_ref.append(best_ref)
        total_best_gen.append(best_gen)
        total_mae_list.append(mae)
        total_edge_sims.extend([e[2] for e in edges])
        total_axis_flags.extend([e[3] for e in edges])
        processed += 1

    # overall single metrics
    def avg(x): return float(np.mean(x)) if x else 0.0
    overall = {
        "precision_node": round(avg(total_p_node), 4),
        "recall_node": round(avg(total_r_node), 4),
        "f1_node": round(avg(total_f1_node), 4),
        "best_sim_ref": round(avg(total_best_ref), 4),
        "best_sim_gen": round(avg(total_best_gen), 4),
        "avg_edge_similarity": round(avg(total_edge_sims), 4),
        "mae_points": round(avg(total_mae_list), 4),
        "axis_match_rate": round((sum(total_axis_flags)/len(total_axis_flags)) if total_axis_flags else 0.0, 4),
        "processed_indices": processed,
        "skipped_indices": skipped,
        "model": args.model,
        "threshold": args.threshold,
        "topk_ref_per_gen": args.topk_ref_per_gen,
        "topk_gen_per_ref": args.topk_gen_per_ref,
        "axis_strict": bool(args.axis_strict),
        "axis_bonus": args.axis_bonus,
        "mutual_topk": bool(args.mutual_topk),
    }

    # --------- p-value computations ---------
    # 1) per-index F1 vs 0.5 (H0: mean(F1) == 0.5)
    pval_f1_ttest = None
    pval_f1_wilcoxon = None
    if len(total_f1_node) >= 2:  # need at least 2 samples for meaningful tests
        f1_arr = np.asarray(total_f1_node, dtype=np.float64)
        try:
            _, pval_f1_ttest = ttest_1samp(f1_arr, popmean=0.5, alternative='two-sided')
            pval_f1_ttest = float(pval_f1_ttest)
        except Exception:
            pval_f1_ttest = None
        try:
            # Wilcoxon on differences (F1 - 0.5)
            diff = f1_arr - 0.5
            # Wilcoxon requires non-zero differences; catch errors when all zeros
            if np.allclose(diff, 0.0):
                pval_f1_wilcoxon = 1.0
            else:
                _, pval_f1_wilcoxon = wilcoxon(diff, alternative='two-sided', zero_method='wilcox')
                pval_f1_wilcoxon = float(pval_f1_wilcoxon)
        except Exception:
            pval_f1_wilcoxon = None

    # 2) edge similarities vs threshold (H0: median(sim - thr) == 0)
    pval_edge_vs_thr = None
    if len(total_edge_sims) >= 10:  # need enough edges
        diffs = np.asarray(total_edge_sims, dtype=np.float64) - float(args.threshold)
        try:
            if np.allclose(diffs, 0.0):
                pval_edge_vs_thr = 1.0
            else:
                _, pval_edge_vs_thr = wilcoxon(diffs, alternative='two-sided', zero_method='wilcox')
                pval_edge_vs_thr = float(pval_edge_vs_thr)
        except Exception:
            pval_edge_vs_thr = None

    overall.update({
        "p_value_f1_ttest": pval_f1_ttest,
        "p_value_f1_wilcoxon": pval_f1_wilcoxon,
        "p_value_edge_sim_vs_threshold_wilcoxon": pval_edge_vs_thr
    })
    # --------- end p-value ---------

    with open(args.overall_json, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    with open(args.overall_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(overall.keys()))
        writer.writeheader()
        writer.writerow(overall)

    print("[SUMMARY] processed:", processed, "skipped:", skipped, "elapsed(s):", round(time.time() - t0, 2))
    print("[WRITE] per-index pairs dir:", args.out_dir)
    print("[WRITE] overall JSON:", args.overall_json)
    print("[WRITE] overall CSV :", args.overall_csv)
    if pval_f1_ttest is not None or pval_f1_wilcoxon is not None:
        print(f"[P-VALUES] F1 vs 0.5 -> t-test: {pval_f1_ttest}, wilcoxon: {pval_f1_wilcoxon}")
    if pval_edge_vs_thr is not None:
        print(f"[P-VALUES] EdgeSim-Threshold -> wilcoxon: {pval_edge_vs_thr}")


if __name__ == "__main__":
    main()
