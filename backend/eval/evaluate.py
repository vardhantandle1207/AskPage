"""
evaluate.py — measures how good AskPage is, stage by stage.

For every question in test_set.json:
    1. Index the sample page through the real ingestion pipeline.
    2. RETRIEVAL QUALITY — run retrieval in three modes (vector only, BM25
       only, hybrid + rerank) and check whether the chunk containing the
       `evidence` string was returned, and at what rank:
           hit@k  = fraction of questions whose evidence chunk is in the top-k
           MRR    = mean of 1/rank of the first evidence chunk (0 if missing)
    3. ANSWER ACCURACY — generate an answer with the full pipeline and ask
       the LLM to judge it against `expected_answer` (YES/NO).
    4. GROUNDING — for questions whose answer is *not* on the page, check
       the model declined instead of inventing something.
    5. LATENCY — p50/p95 for retrieval, rerank and generation.

Everything is written to results.json. Questions the model gets wrong are the
obvious candidates to expand the test set with.

Run from the backend/ folder:   python eval/evaluate.py
Inside Docker:                  docker compose exec backend python eval/evaluate.py
"""

import json
import os
import re
import statistics
import sys
import time
import unicodedata

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm
import rag

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")
# (label, retrieval mode, cross-encoder on/off). The label must say whether
# the reranker ran: a "vector" row measured with the cross-encoder still on is
# really "vector+rerank", and reporting it as "vector" overstates what dense
# retrieval alone achieves.
CONFIGS = [("dense", "vector", False), ("sparse", "bm25", False),
           ("hybrid", "hybrid", False), ("hybrid+rerank", "hybrid", True)]
MODES = [label for label, _, _ in CONFIGS]
NOT_FOUND = "couldn't find that on this page"

JUDGE_PROMPT = (
    "You are grading a student's answer against a reference answer.\n\n"
    "QUESTION: {question}\n"
    "REFERENCE ANSWER: {expected}\n"
    "STUDENT ANSWER: {given}\n\n"
    "Mark it CORRECT if the student answer states the key fact(s) of the "
    "reference answer. Extra detail, different wording, or a longer sentence "
    "are all fine. Mark it WRONG only if a key fact is missing or contradicted.\n"
    "Reply with exactly one word: CORRECT or WRONG."
)

_STOPWORDS = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is",
              "are", "it", "its", "at", "with", "as", "by", "that", "this", "be",
              "should", "only", "meant", "with"}


def _content_words(text: str) -> set[str]:
    """Lowercase, accent-stripped alphanumeric words, minus stopwords."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


def contains_key_facts(expected: str, given: str) -> bool:
    """Deterministic grader: every content word of `expected` appears in `given`."""
    expected_words = _content_words(expected)
    return bool(expected_words) and expected_words <= _content_words(given)


def judge_answer(question: str, expected: str, given: str) -> bool:
    """
    Two-stage grading. A small local model is an unreliable judge (it will
    say "NO" to an answer that plainly contains the expected fact), so we
    first apply an exact key-fact check and only ask the LLM for the
    ambiguous remainder (paraphrases, different word forms).
    """
    given = re.sub(r"\[\d+\]", "", given)          # drop citation markers
    if contains_key_facts(expected, given):
        return True
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, given=given)
    verdict = llm.complete_chat([{"role": "user", "content": prompt}]).strip().upper()
    return verdict.startswith("CORRECT")


def evidence_rank(chunks: list[dict], evidence: str) -> int | None:
    """1-based rank of the first chunk containing `evidence`, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if evidence.lower() in chunk["text"].lower():
            return rank
    return None


def pct(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))], 4)


def main():
    with open(TEST_SET_PATH, encoding="utf-8") as file:
        test_set = json.load(file)

    # Optional: --pages 0,1  runs only those pages and merges into any existing
    # results.json. Generation on CPU is slow, so a 40-question run is easier to
    # do in pieces than in one sitting.
    selected = None
    if "--pages" in sys.argv:
        selected = {int(x) for x in sys.argv[sys.argv.index("--pages") + 1].split(",")}

    cached = {}
    if selected is not None and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as file:
            for record in json.load(file).get("details", []):
                cached[record["question"]] = record

    details = []
    retrieval_hits = {mode: [] for mode in MODES}   # per-question rank or None
    latencies = {"retrieval": [], "rerank": [], "generation": []}

    for page_index, page in enumerate(test_set["pages"]):
        if selected is not None and page_index not in selected:
            for item in page["questions"]:            # carry earlier results through
                if item["question"] in cached:
                    details.append(cached[item["question"]])
            print(f"\nSkipping page {page_index} ({len(details)} cached records carried)")
            continue
        print(f"\nIndexing: {page['title']}")
        summary = rag.index_page(page["url"], page["title"],
                                 text=page.get("text"), blocks=page.get("blocks"))
        print(f"  {summary['num_chunks']} chunks, {summary.get('num_sections', 1)} sections")

        for item in page["questions"]:
            question, expected = item["question"], item["expected_answer"]
            evidence = item.get("evidence")
            record = {"question": question, "expected": expected, "retrieval": {}}

            record["page"] = page["title"]
            # --- retrieval quality, per mode ---
            for label, mode, use_rerank in CONFIGS:
                result = rag.retrieve_chunks(page["url"], question, mode=mode,
                                             rerank=use_rerank)
                rank = evidence_rank(result["chunks"], evidence) if evidence else None
                record["retrieval"][label] = {"rank": rank, "returned": len(result["chunks"])}
                if evidence:
                    retrieval_hits[label].append(rank)
                if label == "hybrid+rerank":
                    hybrid = result

            # --- answer accuracy with the full pipeline ---
            t0 = time.time()
            given = llm.generate_answer(question, hybrid["chunks"], page["title"])
            record["latency"] = {
                "retrieval": hybrid["timings"]["retrieval"],
                "rerank": hybrid["timings"]["rerank"],
                "generation": round(time.time() - t0, 3),
            }

            declined = NOT_FOUND in given.lower()
            if evidence is None:                     # answer is not on the page
                correct = declined
            else:
                correct = (not declined) and judge_answer(question, expected, given)

            record.update({"given": given, "correct": correct, "declined": declined,
                           "should_decline": evidence is None,
                           "cited": "[" in given})
            details.append(record)

            status = "PASS" if correct else "FAIL"
            rank = record["retrieval"]["hybrid+rerank"]["rank"]
            print(f"  [{status}] {question}   (evidence rank: {rank})")
            print(f"         -> {given[:110]}")

    # --- aggregate ---
    def hit_at(ranks, k):
        return round(sum(1 for r in ranks if r and r <= k) / len(ranks), 3) if ranks else None

    def mrr(ranks):
        return round(sum(1 / r for r in ranks if r) / len(ranks), 3) if ranks else None

    # Recomputed from `details` rather than accumulated during the loop, so a
    # partial run (--pages) that carries cached records through still aggregates
    # over all 40 questions.
    retrieval_hits = {mode: [d["retrieval"][mode]["rank"] for d in details
                             if not d["should_decline"] and mode in d["retrieval"]]
                      for mode in MODES}
    latencies = {name: [d["latency"][name] for d in details if "latency" in d]
                 for name in ("retrieval", "rerank", "generation")}
    retrieval_summary = {
        mode: {"hit@1": hit_at(ranks, 1), "hit@3": hit_at(ranks, 3),
               "hit@5": hit_at(ranks, 5), "mrr": mrr(ranks)}
        for mode, ranks in retrieval_hits.items()
    }
    answerable = [d for d in details if not d["should_decline"]]
    unanswerable = [d for d in details if d["should_decline"]]
    summary = {
        "questions": len(details),
        "answer_accuracy": round(sum(d["correct"] for d in answerable) / len(answerable), 3) if answerable else None,
        "false_answer_rate": round(sum(not d["declined"] for d in unanswerable) / len(unanswerable), 3) if unanswerable else None,
        "citation_rate": round(sum(d["cited"] for d in answerable if not d["declined"]) /
                               max(1, sum(not d["declined"] for d in answerable)), 3),
        "retrieval": retrieval_summary,
        "latency_seconds": {name: {"p50": pct(v, 0.5), "p95": pct(v, 0.95)}
                            for name, v in latencies.items()},
        "provider": llm.LLM_PROVIDER, "model": llm.MODEL_NAME,
        "reranker": rag.RERANKER_MODEL_NAME if rag.reranker else None,
    }

    print("\n========== RESULTS ==========")
    print(f"Questions:          {summary['questions']}")
    print(f"Answer accuracy:    {summary['answer_accuracy']:.0%}  (judge = YES, answerable questions)")
    print(f"False-answer rate:  {summary['false_answer_rate']:.0%}  (answered when it should have declined)")
    print(f"Citation rate:      {summary['citation_rate']:.0%}")
    print("\nRetrieval (evidence chunk found):")
    print(f"  {'mode':<10}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'MRR':>7}")
    for mode, m in retrieval_summary.items():
        print(f"  {mode:<10}{m['hit@1']:>7}{m['hit@3']:>7}{m['hit@5']:>7}{m['mrr']:>7}")
    print("\nLatency p50 / p95 (s):")
    for name, v in summary["latency_seconds"].items():
        print(f"  {name:<12}{v['p50']} / {v['p95']}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump({"summary": summary, "details": details}, file, indent=2, ensure_ascii=False)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
