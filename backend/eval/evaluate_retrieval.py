"""
evaluate_retrieval.py — measures the retrieval funnel only. No LLM needed.

Runs every question in test_set.json through four configurations so each
stage of the funnel can be attributed:

    dense          FAISS vector search alone
    sparse         BM25 alone
    hybrid         dense + sparse fused with Reciprocal Rank Fusion
    hybrid+rerank  the above, reranked by the MiniLM cross-encoder

For each it reports, over the questions that have a labelled evidence chunk:

    hit@k  fraction of questions whose evidence chunk is in the top k
    MRR    mean of 1/rank of the first evidence chunk (0 if not returned)

Latency is wall-clock on this machine, split into retrieval and rerank.
Answer accuracy, citation rate and false-answer rate need a generator and
live in evaluate.py — run that with your own Ollama or Groq.

Run from backend/:   python eval/evaluate_retrieval.py
"""

import json
import os
import statistics
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results_retrieval.json")

# (label, retrieval mode, cross-encoder on/off)
CONFIGS = [
    ("dense", "vector", False),
    ("sparse", "bm25", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]
TOP_K = 5
REPEATS = 3          # repeat each timed call; keeps one cold run from skewing p95


def evidence_rank(chunks, evidence):
    """1-based rank of the first chunk containing `evidence`, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if evidence.lower() in chunk["text"].lower():
            return rank
    return None


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
    return round(ordered[index], 4)


def main():
    with open(TEST_SET_PATH, encoding="utf-8") as file:
        test_set = json.load(file)

    ranks = {label: [] for label, _, _ in CONFIGS}
    timings = {label: {"retrieval": [], "rerank": [], "total": []} for label, _, _ in CONFIGS}
    per_page = {}
    details = []
    chunk_stats = []

    for page in test_set["pages"]:
        summary = rag.index_page(page["url"], page["title"], blocks=page.get("blocks"),
                                 text=page.get("text"))
        chunk_stats.append((page["title"], summary["num_chunks"],
                            summary.get("num_sections"), summary.get("index_time_seconds")))
        print(f"\n{page['genre']}: {page['title'][:55]}")
        print(f"  {summary['num_chunks']} chunks / {summary.get('num_sections')} sections "
              f"in {summary.get('index_time_seconds')}s")

        page_ranks = {label: [] for label, _, _ in CONFIGS}

        for item in page["questions"]:
            question = item["question"]
            evidence = item.get("evidence")
            record = {"page": page["title"], "question": question,
                      "answerable": evidence is not None, "ranks": {}}

            for label, mode, use_rerank in CONFIGS:
                # timed repeats
                best = None
                for _ in range(REPEATS):
                    start = time.perf_counter()
                    result = rag.retrieve_chunks(page["url"], question, top_k=TOP_K,
                                                 mode=mode, rerank=use_rerank)
                    elapsed = time.perf_counter() - start
                    if best is None or elapsed < best[0]:
                        best = (elapsed, result)
                elapsed, result = best

                timings[label]["retrieval"].append(result["timings"]["retrieval"])
                timings[label]["rerank"].append(result["timings"]["rerank"])
                timings[label]["total"].append(round(elapsed, 4))

                if evidence is not None:
                    rank = evidence_rank(result["chunks"], evidence)
                    ranks[label].append(rank)
                    page_ranks[label].append(rank)
                    record["ranks"][label] = rank

            if evidence is not None:
                marks = " ".join(
                    f"{label}={record['ranks'][label] or '-'}" for label, _, _ in CONFIGS)
                print(f"  {marks:<52} {question[:60]}")
            details.append(record)

        per_page[page["title"]] = {
            "genre": page["genre"],
            "chunks": summary["num_chunks"],
            "hit@1": {label: hit_at(page_ranks[label], 1) for label, _, _ in CONFIGS},
        }

    # ---------------- aggregate ----------------
    table = {}
    for label, _, _ in CONFIGS:
        r = ranks[label]
        table[label] = {
            "hit@1": hit_at(r, 1), "hit@3": hit_at(r, 3), "hit@5": hit_at(r, 5),
            "mrr": mrr(r),
            "latency_ms": {
                "p50": ms(percentile(timings[label]["total"], 0.5)),
                "p95": ms(percentile(timings[label]["total"], 0.95)),
                "mean": ms(statistics.mean(timings[label]["total"])),
            },
            "retrieval_ms_p50": ms(percentile(timings[label]["retrieval"], 0.5)),
            "rerank_ms_p50": ms(percentile(timings[label]["rerank"], 0.5)),
        }

    answerable = sum(1 for d in details if d["answerable"])
    print("\n" + "=" * 78)
    print(f"RETRIEVAL — {len(details)} questions "
          f"({answerable} answerable, {len(details) - answerable} unanswerable), "
          f"{len(test_set['pages'])} real pages, top_k={TOP_K}")
    print("=" * 78)
    print(f"{'config':<16}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}"
          f"{'p50 ms':>10}{'p95 ms':>10}")
    for label, _, _ in CONFIGS:
        m = table[label]
        print(f"{label:<16}{m['hit@1']:>8.3f}{m['hit@3']:>8.3f}{m['hit@5']:>8.3f}"
              f"{m['mrr']:>8.3f}{m['latency_ms']['p50']:>10.1f}{m['latency_ms']['p95']:>10.1f}")

    print("\nhit@1 by page (hybrid+rerank):")
    for title, info in per_page.items():
        print(f"  {info['genre']:<22}{info['hit@1']['hybrid+rerank']:>6.2f}   "
              f"{info['chunks']:>3} chunks   {title[:40]}")

    misses = [d for d in details if d["answerable"] and
              (d["ranks"].get("hybrid+rerank") is None or d["ranks"]["hybrid+rerank"] > 1)]
    print(f"\nnot ranked first by hybrid+rerank: {len(misses)}/{answerable}")
    for d in misses:
        print(f"  rank {d['ranks'].get('hybrid+rerank')}: {d['question'][:70]}")

    output = {
        "setup": {
            "questions": len(details), "answerable": answerable,
            "unanswerable": len(details) - answerable,
            "pages": [{"title": t, "chunks": c, "sections": s, "index_seconds": i}
                      for t, c, s, i in chunk_stats],
            "top_k": TOP_K, "repeats": REPEATS,
            "embedding_model": rag.EMBEDDING_MODEL_NAME,
            "reranker_model": rag.RERANKER_MODEL_NAME,
            "machine": "evaluation sandbox CPU — re-run locally for your own latency",
        },
        "retrieval": table,
        "per_page": per_page,
        "details": details,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
    print(f"\nSaved to {RESULTS_PATH}")


def hit_at(rank_list, k):
    if not rank_list:
        return 0.0
    return round(sum(1 for r in rank_list if r and r <= k) / len(rank_list), 3)


def mrr(rank_list):
    if not rank_list:
        return 0.0
    return round(sum(1 / r for r in rank_list if r) / len(rank_list), 3)


def ms(seconds):
    return round((seconds or 0) * 1000, 1)


if __name__ == "__main__":
    main()
