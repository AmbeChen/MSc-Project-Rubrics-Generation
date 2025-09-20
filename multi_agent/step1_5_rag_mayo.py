# === file: multi_steps/step1_5_rag_mayo.py (part 1/5) ===
import os, re, json, time, argparse, html
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "RubricsGen/0.2 (+https://example.local)"

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def append_jsonl(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def html_to_text(html_str: str) -> str:
    if not html_str:
        return ""
    soup = BeautifulSoup(html_str, "html.parser")
    for tag in soup(["script","style","noscript","header","footer","nav","aside"]):
        tag.extract()
    text = soup.get_text(" ")
    
    # strip common boilerplate
    for pat in ["Request Appointment", "Patient & Visitor Guide", "Search Menu", "Care at Mayo Clinic"]:
        text = text.replace(pat, " ")
        
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



# === (part 2/5) ===
_STOP = set("a an the of for to in on with about into by at from and or if is are was were be as than that this those these which who whom whose".split())

def sent_tokenize(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def para_split(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras

def key_terms_from_query(q: str) -> List[str]:
    toks = [t.lower() for t in re.findall(r"[a-z0-9\-]+", q)]
    return [t for t in toks if t not in _STOP and len(t) > 1]

def score_sentence(s: str, key_terms: List[str]) -> int:
    s_low = s.lower()
    return sum(1 for t in key_terms if re.search(rf"\b{re.escape(t)}\b", s_low))

def expand_anchor_to_paragraph(full_text: str,
                               anchor_sent: str,
                               window: int = 2,
                               target_words: int = 150) -> str:
    # First, find anchor points in the paragraph, If it can be found, make a sentence window within this paragraph; otherwise, directly use the global sentence list for the window
    paras = para_split(full_text)
    for p in paras:
        if anchor_sent in p:
            sents = sent_tokenize(p)
            try:
                idx = sents.index(anchor_sent)
            except ValueError:
                idx = max((i for i, ss in enumerate(sents) if anchor_sent[:40] in ss), default=0)
            start = max(0, idx - window); end = min(len(sents), idx + window + 1)
            chosen = " ".join(sents[start:end]).strip()
            break
    else:
        sents = sent_tokenize(full_text)
        try:
            idx = sents.index(anchor_sent)
        except ValueError:
            idx = 0
        start = max(0, idx - window); end = min(len(sents), idx + window + 1)
        chosen = " ".join(sents[start:end]).strip()

    words = chosen.split()
    if len(words) > target_words:
        chosen = " ".join(words[:target_words])
    return re.sub(r"\s+", " ", chosen).strip()




# === (part 3/5) ===
def serper_search_mayo(query: str, api_key: str, topn: int = 5) -> List[Dict[str, str]]:
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": f"site:mayoclinic.org {query}", "num": topn}
    r = requests.post(url, headers=headers, json=payload, timeout=25)
    r.raise_for_status()
    data = r.json()
    out = []
    for it in data.get("organic", [])[:topn]:
        link = it.get("link")
        if link and "mayoclinic.org" in link:
            out.append({
                "title": it.get("title", "") or urlparse(link).netloc,
                "url": link,
                "snippet": it.get("snippet", "") or ""
            })
    return out

def http_get(url: str, timeout: int = 20) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r.text



# === (part 4/5) ===
def choose_best_result(results: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not results: 
        return None
    # Simple strategy: Prioritize Mayo pages with clear urls/titles; Here, the first one is sufficient
    return results[0]

def passage_from_query(query: str,
                       api_key: str,
                       target_words: int = 150,
                       sent_window: int = 2,
                       sleep_between: float = 0.4) -> Optional[Dict[str, str]]:
    hits = serper_search_mayo(query, api_key, topn=5)
    best = choose_best_result(hits)
    if not best:
        return None
    url = best["url"]
    snippet = best.get("snippet","").strip()

    try:
        html_str = http_get(url)
        text = html_to_text(html_str)
    except Exception:
        text = ""

    if not text and snippet:
        # Fallback: Use snippet to fill a small segment
        txt = re.sub(r"\s+", " ", snippet).strip()
        words = txt.split()
        txt = " ".join(words[:target_words])
        return {"source": best.get("title") or urlparse(url).netloc,
                "url": url, "text": txt}

    if not text:
        return None

    # Select "anchor sentence"
    sents = sent_tokenize(text)
    anchor = None
    if snippet:
        sn_terms = key_terms_from_query(snippet)
        best_sc, best_sent = -1, None
        for ss in sents:
            sc = score_sentence(ss, sn_terms)
            if sc > best_sc:
                best_sc, best_sent = sc, ss
        anchor = best_sent
    if not anchor:
        q_terms = key_terms_from_query(query)
        best_sc, best_sent = -1, None
        for ss in sents:
            sc = score_sentence(ss, q_terms)
            if sc > best_sc:
                best_sc, best_sent = sc, ss
        anchor = best_sent or (sents[0] if sents else "")

    para = expand_anchor_to_paragraph(text, anchor, window=sent_window, target_words=target_words)
    time.sleep(sleep_between)
    if not para:
        return None

    return {
        "source": best.get("title") or urlparse(url).netloc,
        "url": url,
        "text": para
    }

def jaccard_sim(a: str, b: str) -> float:
    A = set(re.findall(r"[a-z0-9]+", a.lower()))
    B = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def build_bundle_for_row(index: int,
                         queries: List[str],
                         api_key: str,
                         per_query_words: int = 150,
                         global_cap: Optional[int] = None,
                         dedupe: bool = False,
                         dedupe_threshold: float = 0.8,
                         global_token_budget: int = 1400,
                         approx_tokens_per_word: float = 1.2) -> Dict[str, Any]:
    # For each query, attempt to retrieve 1 passage
    passages: List[Dict[str, str]] = []
    for i, q in enumerate(queries):
        p = passage_from_query(q, api_key=api_key,
                               target_words=per_query_words,
                               sent_window=2)
        if p:
            p["id"] = f"p{i+1}"
            passages.append(p)

    # Light cross-query deduplication (optional)
    if dedupe and len(passages) >= 2:
        kept = []
        for p in passages:
            keep = True
            for q in kept:
                if jaccard_sim(p["text"], q["text"]) >= dedupe_threshold:
                    # If the two segments are similar: keep the one with more/longer information
                    if len(p["text"]) > len(q["text"]):
                        q.update(p)
                    keep = False
                    break
            if keep:
                kept.append(p)
        passages = kept

    # Global cap (default = #queries, in line with "at least 1 passage per query" strategy)
    if global_cap is None:
        global_cap = len(queries)
    passages = passages[:global_cap]

    # Global token budget (approx: 1 word ≈ 1.2 tokens)
    max_words = int(global_token_budget / approx_tokens_per_word)
    total_words = sum(len(p["text"].split()) for p in passages)
    if total_words > max_words and total_words > 0:
        ratio = max_words / total_words
        for p in passages:
            words = p["text"].split()
            new_len = max(60, int(len(words) * ratio))
            p["text"] = " ".join(words[:new_len]).strip()

    return {"index": index, "queries": queries, "passages": passages}


# === (part 5/5) ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals_path", type=str, default="multi_steps_v2/outputs/step1_signals_all.jsonl")
    ap.add_argument("--out_all_path", type=str, default="multi_steps_v2/outputs/step1_5_rag_all.jsonl")
    ap.add_argument("--serper_api_key", type=str, default=os.getenv("SERPER_API_KEY"))
    ap.add_argument("--per_query_words", type=int, default=150)
    ap.add_argument("--global_cap", type=int, default=None)
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--dedupe_threshold", type=float, default=0.8)
    ap.add_argument("--global_token_budget", type=int, default=1400)
    ap.add_argument("--sleep_between", type=float, default=0.4)
    ap.add_argument("--start_index", type=int, default=None)
    ap.add_argument("--end_index", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not args.serper_api_key:
        raise SystemExit("Missing Serper API key. Set --serper_api_key or SERPER_API_KEY.")

    rows = load_jsonl(args.signals_path)

    # Only process the specified range
    if args.start_index is not None and args.end_index is not None:
        rows = [row for row in rows
                if args.start_index <= int(row.get("index", 0)) < args.end_index]

    #if os.path.exists(args.out_all_path):
    #    os.remove(args.out_all_path)

    for row in rows:
        idx = int(row.get("index", 0))
        queries = row.get("queries") or []
        bundle = build_bundle_for_row(
            index=idx,
            queries=queries,
            api_key=args.serper_api_key,
            per_query_words=args.per_query_words,
            global_cap=args.global_cap,
            dedupe=args.dedupe,
            dedupe_threshold=args.dedupe_threshold,
            global_token_budget=args.global_token_budget
        )
        append_jsonl(bundle, args.out_all_path)
        time.sleep(args.sleep_between)

    print(f"[OK] wrote RAG bundles -> {args.out_all_path} ({len(rows)} lines)")

if __name__ == "__main__":
    main()

