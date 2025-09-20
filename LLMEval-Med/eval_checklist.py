# eval_checklist.py
import os, json, argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_reference(jsonl_path):
    refs = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            refs[str(i)] = obj.get("checklist", [])
    return refs

def load_generated(gen_dir):
    gens = {}
    for fn in os.listdir(gen_dir):
        if fn.startswith("rubrics_") and fn.endswith(".json"):
            idx = fn.split("_")[1].split(".")[0]
            with open(os.path.join(gen_dir, fn), "r", encoding="utf-8") as f:
                gens[idx] = json.load(f)
    return gens

def cosine_matrix(A, B):
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    return np.matmul(A, B.T).astype(np.float32)

def evaluate(gen_dir, ref_jsonl, model_name="BAAI/bge-m3", threshold=0.5): # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    # load data
    refs = load_reference(ref_jsonl)
    gens = load_generated(gen_dir)

    model = SentenceTransformer(model_name)
    results = {}

    total_p, total_r, total_f1 = [], [], []

    for idx, ref_items in tqdm(refs.items()):
        gen_items = gens.get(idx, [])
        if not ref_items or not gen_items:
            continue

        # encode
        ref_emb = model.encode(ref_items, convert_to_numpy=True, normalize_embeddings=True)
        gen_emb = model.encode(gen_items, convert_to_numpy=True, normalize_embeddings=True)

        sim = cosine_matrix(gen_emb, ref_emb)
        matched_gen, matched_ref = set(), set()

        for i in range(sim.shape[0]):
            j = int(np.argmax(sim[i]))
            if sim[i, j] >= threshold:
                matched_gen.add(i)
                matched_ref.add(j)

        p = len(matched_gen) / len(gen_items) if gen_items else 0
        r = len(matched_ref) / len(ref_items) if ref_items else 0
        f1 = 2*p*r / (p+r) if (p+r) > 0 else 0

        results[idx] = {"precision": p, "recall": r, "f1": f1}
        total_p.append(p)
        total_r.append(r)
        total_f1.append(f1)

    overall = {
        "precision": float(np.mean(total_p)),
        "recall": float(np.mean(total_r)),
        "f1": float(np.mean(total_f1)),
    }

    return results, overall

from scipy.stats import ttest_rel, wilcoxon

def evaluate_with_pvalue(all_scores_ref, all_scores_gen):
    """
    Input：
      all_scores_ref: list[float] Refer to the score of the checklist (ideally, it can be set to 1 or gold match)
      all_scores_gen: list[float] Generate the similarity/coverage score of the checklist
    Output：
      t-test and Wilcoxon results
    """

    # Paired t-test (assuming the scores are approximately normally distributed)
    t_stat, p_ttest = ttest_rel(all_scores_ref, all_scores_gen)

    # Wilcoxon signed-rank test (non-parametric, suitable for unknown distributions)
    try:
        w_stat, p_wilcoxon = wilcoxon(all_scores_ref, all_scores_gen)
    except ValueError:
        # report an error if sample size is too small
        w_stat, p_wilcoxon = None, None

    print("\n===== Statistical Significance Test =====")
    print(f"Paired t-test: t = {t_stat:.4f}, p = {p_ttest:.4g}")
    if p_wilcoxon is not None:
        print(f"Wilcoxon test: W = {w_stat:.4f}, p = {p_wilcoxon:.4g}")
    else:
        print("Wilcoxon test: not applicable")

    return {
        "t-test": {"t": t_stat, "p": p_ttest},
        "wilcoxon": {"W": w_stat, "p": p_wilcoxon}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--ref_jsonl", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    results, overall = evaluate(args.gen_dir, args.ref_jsonl)

    # Collect all f1 scores and construct a list of "reference =1"
    all_f1 = [v["f1"] for v in results.values()]
    all_ref = [1.0] * len(all_f1)

    stats = evaluate_with_pvalue(all_ref, all_f1)

    overall["p_value_ttest"] = stats["t-test"]["p"]
    overall["p_value_wilcoxon"] = stats["wilcoxon"]["p"]

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "per_index": results}, f, ensure_ascii=False, indent=2)

    print("✅ Done. Overall:", overall)
