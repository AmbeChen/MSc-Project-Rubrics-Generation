# eval_checklist_summary.py
import os, json, argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.stats import ttest_rel, wilcoxon

def load_reference(jsonl_path):
    refs = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            refs[str(i)] = obj.get("checklist", []) or []
    return refs

def load_generated_from_summary(jsonl_path):
    gens = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            idx = str(obj.get("index"))
            # Step3 summary may contain "rubrics" or "final"
            items = []
            if "rubrics" in obj and isinstance(obj["rubrics"], list):
                for it in obj["rubrics"]:
                    if isinstance(it, dict):
                        c = it.get("criterion", "")
                        if c: items.append(c.strip())
                    elif isinstance(it, str):
                        items.append(it.strip())
            elif "final" in obj and isinstance(obj["final"], list):
                for it in obj["final"]:
                    if isinstance(it, dict):
                        c = it.get("criterion", "")
                        if c: items.append(c.strip())
            gens[idx] = items
    return gens

def cosine_matrix(A, B):
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    return (A @ B.T).astype(np.float32)

def evaluate(summary_jsonl, ref_jsonl, model_name="BAAI/bge-m3", threshold=0.5): #sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    refs = load_reference(ref_jsonl)
    gens = load_generated_from_summary(summary_jsonl)

    model = SentenceTransformer(model_name)
    results = {}
    total_p, total_r, total_f1 = [], [], []

    for idx, ref_items in tqdm(refs.items(), desc="Evaluating"):
        gen_items = gens.get(idx, [])
        if not ref_items or not gen_items:
            continue

        ref_emb = model.encode(ref_items, convert_to_numpy=True, normalize_embeddings=True)
        gen_emb = model.encode(gen_items, convert_to_numpy=True, normalize_embeddings=True)

        sim = cosine_matrix(gen_emb, ref_emb)
        matched_gen, matched_ref = set(), set()

        for i in range(sim.shape[0]):
            j = int(np.argmax(sim[i]))
            if sim[i, j] >= threshold:
                matched_gen.add(i)
                matched_ref.add(j)

        p = len(matched_gen) / len(gen_items) if gen_items else 0.0
        r = len(matched_ref) / len(ref_items) if ref_items else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        results[idx] = {"precision": p, "recall": r, "f1": f1}
        total_p.append(p); total_r.append(r); total_f1.append(f1)

    overall = {
        "precision": float(np.mean(total_p)) if total_p else 0.0,
        "recall": float(np.mean(total_r)) if total_r else 0.0,
        "f1": float(np.mean(total_f1)) if total_f1 else 0.0,
    }
    return results, overall

def evaluate_with_pvalue(all_scores_ref, all_scores_gen):
    t_stat, p_ttest = ttest_rel(all_scores_ref, all_scores_gen)
    try:
        w_stat, p_wilcoxon = wilcoxon(all_scores_ref, all_scores_gen)
    except ValueError:
        w_stat, p_wilcoxon = None, None
    return {"t-test": {"t": t_stat, "p": p_ttest},
            "wilcoxon": {"W": w_stat, "p": p_wilcoxon}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_jsonl", type=str, required=True, help="Step3 reviewed summary JSONL.")
    parser.add_argument("--ref_jsonl", type=str, required=True, help="Reference JSONL with checklists.")
    parser.add_argument("--out_json", type=str, required=True, help="Output metrics JSON.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    results, overall = evaluate(args.summary_jsonl, args.ref_jsonl, threshold=args.threshold)

    all_f1 = [v["f1"] for v in results.values()]
    all_ref = [1.0] * len(all_f1)
    stats = evaluate_with_pvalue(all_ref, all_f1)

    overall["p_value_ttest"] = stats["t-test"]["p"]
    overall["p_value_wilcoxon"] = stats["wilcoxon"]["p"]

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "per_index": results}, f, ensure_ascii=False, indent=2)

    print("✅ Done. Overall:", overall)
