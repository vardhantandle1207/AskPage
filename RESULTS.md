# AskPage, evaluation results

Retrieval measured on **40 questions across 4 real web pages** (32 answerable,
8 deliberately unanswerable). Raw output: `backend/eval/results_retrieval.json`.
Reproduce with `python eval/evaluate_retrieval.py` from `backend/`.

## Test set

| Page | Genre | Chunks | Sections | Questions |
|---|---|---|---|---|
| `docs.python.org/3/library/venv.html` | technical reference | 23 | 5 | 10 |
| `developer.mozilla.org/.../Using_Fetch` | developer tutorial | 23 | 16 | 10 |
| `en.wikipedia.org/wiki/Calvin_cycle` | science encyclopedia | 21 | 10 | 10 |
| `en.wikipedia.org/wiki/Harappan_architecture` | history article | 21 | 15 | 10 |

Each answerable question carries an `evidence` substring that must appear in a
retrieved chunk; each evidence string was verified to occur exactly once in its
page, so a hit is unambiguous. Two questions per page are unanswerable from that
page, plausible, adjacent, and deliberately not covered (e.g. "Has the Indus
script been deciphered?" on a page about Harappan architecture).

## Retrieval funnel, what each stage buys

Top-k = 5, 32 answerable questions.

| Config | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| dense (FAISS, bge-small) | 0.625 | 0.781 | 0.875 | 0.717 |
| sparse (BM25) | 0.781 | 1.000 | 1.000 | 0.875 |
| hybrid (RRF fusion) | 0.688 | 0.875 | 0.938 | 0.786 |
| **hybrid + cross-encoder rerank** | **0.844** | **1.000** | **1.000** | **0.917** |

Reading this honestly:

- **The full funnel wins**: hit@1 0.844, MRR 0.917, and the evidence chunk is in
  the top 5 handed to the model for **100%** of answerable questions.
- **Dense alone is the weakest stage** (0.625). It misses questions that turn on
  exact tokens, `--prompt`, `--without-pip`, `pyvenv.cfg`, the `1x2x4` brick
  ratio, which is precisely the failure mode BM25 covers.
- **BM25 alone beats naive RRF fusion** on this set (0.781 vs 0.688). Fusion
  pulls weak dense candidates up the list and displaces good lexical hits.
  Fusion is a *recall* step, not a precision step; it is only worth it because
  the cross-encoder reorders afterwards. This is the result I'd expect to be
  asked about, and it is the argument for keeping the reranker rather than
  shipping fusion alone.
- 5 of 32 answerable questions are not ranked first by the full funnel; all 5
  are ranked 2nd or 3rd, so the model still receives the evidence.

### hit@1 by page (hybrid + rerank)

| Genre | hit@1 |
|---|---|
| developer tutorial (MDN) | 1.00 |
| technical reference (Python docs) | 0.88 |
| science encyclopedia (Wikipedia) | 0.75 |
| history article (Wikipedia) | 0.75 |

Prose-heavy encyclopedia pages are the hard case: adjacent chunks restate the
same facts, so the "right" chunk is often a near-tie with a plausible neighbour.

## Latency

Measured on an Apple Silicon MacBook Air, CPU only, no GPU.

| Config | total p50 | total p95 |
|---|---|---|
| dense | 6.3 ms | 7.9 ms |
| sparse | 0.0 ms | 0.0 ms |
| hybrid | 6.1 ms | 7.2 ms |
| hybrid + rerank | 143.4 ms | 192.6 ms |

Search itself is about 6 ms, almost all of it spent embedding the query. BM25 is
free by comparison. The cross-encoder is the expensive part: scoring roughly 24
candidate pairs costs about 137 ms. That is the price of going from hit@1 0.688
to 0.844, and it is worth it here because the whole pipeline still answers in
well under a second before the model starts writing.

Generation is the slow step, not retrieval: 7.2 s p50 and 13.1 s p95 for a full
answer from llama3.2 3B running locally. The UI streams tokens, so the user sees
the first words long before that.

## Answer quality

From `eval/evaluate.py`, same 40 questions, llama3.2 3B via Ollama.

| Metric | Result |
|---|---|
| Answer accuracy (32 answerable) | 100% |
| Citation rate | 97% |
| False-answer rate (8 unanswerable) | 25% (2 of 8) |

Two caveats worth stating plainly.

The 100% is softer than it looks. Grading runs in two stages: a deterministic
check that every content word of the reference answer appears in the output, and
an LLM judge only for what that misses. Most questions passed at stage one, so
the number means "the model stated the key fact", not "the answer is good".

The 25% is the honest problem. Two unanswerable questions got answered instead of
declined. Looking at the rerank scores, answerable questions score a median of
6.23 but go as low as -7.30, while unanswerable ones reach as high as 1.93. The
two distributions overlap, so no score threshold separates them cleanly. On top of
that, `MIN_KEEP = 2` forces two passages through even when everything is below the
threshold, which is how a question scoring -7.82, the lowest of all 40, still got
an answer.

## Limitations

- 40 questions on 4 pages. Enough to rank the configurations; not enough to
  separate 0.84 from 0.88 with confidence.
- Questions and evidence labels were written by one author against these four
  pages, so they reflect what this pipeline chunks well.
- All four pages are clean, well-structured English documents. Marketing pages,
  SPAs and heavy-JS sites are not represented.
- Retrieval only. Answer quality numbers require the generation run.
