
# AskPage: Context-Aware Browser Assistant

A Chrome extension that lets you ask questions about the web page you are reading. It runs a small Retrieval-Augmented Generation (RAG) pipeline: the page is split into chunks, embedded, searched with FAISS, and the best chunks are handed to a LLaMA model that streams the answer back into a side panel.

Runs entirely on your machine: the backend in Docker (or plain uvicorn), the LLM via Ollama. Free, offline, no keys. Groq's hosted API can be swapped in with one env var if you want faster answers.

---

<img width="1470" height="881" alt="Screenshot 2026-09-21 at 3 50 44 AM" src="https://github.com/user-attachments/assets/78275b38-e35c-439e-ab72-34a4ecb8684e" />


## 1. Project overview

Long documentation pages, research articles and blog posts are slow to search by hand. AskPage adds a side panel to Chrome where you type a question and get an answer grounded in *that page only*, along with the passages it used. If the answer is not on the page, it says so instead of guessing.

The project has two halves:

| Half | What it is | Language |
|------|------------|----------|
| `extension/` | Chrome side-panel UI. Extracts the page as structured blocks, sends it to the backend, renders streamed answers with clickable citations. | JavaScript |
| `backend/` | FastAPI server. Ingestion, hybrid search (FAISS + BM25), cross-encoder reranking, cited generation, tracing and metrics. | Python |

---

## 2. Features

- Ask questions about any normal web page from a side panel
- **Smart ingestion**: the page is parsed into headings, paragraphs, lists, code and tables; cleaned of boilerplate; split at sentence boundaries under its section headings
- **Hybrid search + reranking**: FAISS semantic search and BM25 keyword search fused with RRF, then a cross-encoder reranks and drops noise
- **Citations**: every claim carries a `[n]` chip; click it (or the source card) to scroll to and highlight that passage on the page, labelled with its section
- Says "I couldn't find that on this page" when the page doesn't contain the answer
- **Evaluation & observability**: per-request traces to JSONL, `/metrics` (latency percentiles, grounded rate, error count), and an eval harness reporting hit@k / MRR per funnel stage plus judged answer accuracy
- **Freshness**: pages carry a content hash, so an unchanged page reuses its index and a changed page is rebuilt automatically
- Streams the answer token by token; switchable LLM (Ollama local / Groq cloud) via one env var
- Dockerised backend, one command to build and run

---

## 3. Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| Extension | Chrome Manifest V3, Side Panel API, vanilla JS | No build step, easy to read |
| API server | FastAPI + Uvicorn | Minimal code, automatic validation, async streaming |
| Embeddings | `sentence-transformers` with `BAAI/bge-small-en-v1.5` | Small (33M params), fast on CPU, strong retrieval quality |
| Vector search | FAISS (`IndexFlatIP`) | Exact nearest-neighbour search; a page has only a few hundred chunks so no approximation needed |
| Keyword search | `rank-bm25` (BM25Okapi) | Catches exact tokens (error codes, flags, names) that embeddings blur |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reads question + chunk together; far sharper relevance than either retriever; ~137 ms for ~24 candidates on CPU |
| LLM | LLaMA 3.2 3B via Ollama **or** gpt-oss-20b via Groq | Ollama: fully offline, no key. Groq: free tier, ~1000 tok/s |
| Packaging | Docker + docker-compose | Reproducible build, one command to run, restarts on crash |
| Streaming | NDJSON over HTTP `StreamingResponse` | Simplest way to stream without WebSockets |
| Evaluation | hit@k / MRR per retrieval mode + LLM-as-judge | Shows what each stage adds, not just an overall number |
| Observability | JSONL traces + `/metrics` | Latency per stage and grounded rate over time, without a dashboard dependency |

---

## 4. Project structure

```
askpage/
├── README.md
├── EXPLANATION.md          # step-by-step walkthrough of every file
├── docker-compose.yml      # one-command build & run
├── .gitignore
├── backend/
│   ├── main.py             # FastAPI routes, rate limiting, tracing
│   ├── ingest.py           # clean → sections → sentence-aware chunks → dedupe
│   ├── rag.py              # FAISS + BM25 hybrid search, RRF, cross-encoder rerank
│   ├── llm.py              # cited prompt + Groq/Ollama streaming
│   ├── observability.py    # per-request traces, /metrics
│   ├── Dockerfile          # container image (models baked in)
│   ├── .env.example        # config template (provider, key, tuning)
│   ├── requirements.txt
│   ├── data/               # traces.jsonl (git-ignored, docker volume)
│   └── eval/
│       ├── test_set.json   # 40 questions over 4 real pages + evidence strings
│       ├── evaluate.py     # retrieval metrics + judged answer accuracy (needs an LLM)
│       ├── evaluate_retrieval.py  # retrieval metrics only, no LLM needed
│       └── results_retrieval.json # measured output
└── extension/
    ├── manifest.json       # extension config and permissions
    ├── background.js       # opens the side panel on icon click
    ├── sidepanel.html      # UI layout
    ├── sidepanel.css       # UI styling
    └── sidepanel.js        # extract blocks → index → ask → citations
```

---

## 5. How it works

```
 Chrome tab                Side panel (sidepanel.js)             Backend (FastAPI)
 ──────────                ─────────────────────────             ─────────────────
 page DOM ──executeScript─▶ extractPageContent()
                            {url, title, blocks[], text}
                                    │ POST /index
                                    ▼
                                                        ingest.ingest()          ── 1. SMART INGESTION
                                                        ├─ clean: NFKC, whitespace, boilerplate lines
                                                        ├─ parse: blocks → sections by heading path
                                                        ├─ split: ~220-word chunks at sentence ends,
                                                        │         2-sentence overlap, code/tables intact
                                                        └─ dedupe repeated chunks
                                                        rag.index_page()         ── 5. FRESHNESS
                                                        ├─ content hash unchanged? → cached
                                                        ├─ changed? → re-index ("refreshed")
                                                        ├─ FAISS index (bge-small vectors)
                                                        └─ BM25 index (tokens)
                                    ◀── {num_chunks, num_sections, cached, refreshed}

 user asks ───▶ re-extract, hash, re-index if page changed
                POST /ask {url, question}
                                                        rag.retrieve_chunks()    ── 2. HYBRID + RERANK
                                                        ├─ vector top-12  ┐
                                                        ├─ BM25   top-12  ┴─ RRF fusion → ~20 candidates
                                                        ├─ cross-encoder scores (question, chunk) pairs
                                                        └─ drop below threshold, keep top-5
                                                        llm.stream_answer()      ── 3. CITATIONS
                                                        ├─ sources numbered [n] + section labels
                                                        └─ model must cite [n] per claim
                                                        observability.Trace      ── 4. OBSERVABILITY
                                                        └─ timings, chunk ids, grounded? → traces.jsonl
                ◀── NDJSON: {sources, trace_id} → {token}… → {done, trace_id}
                renders tokens live; [n] → chips that highlight the passage on the page
```

Two phases, both triggered by the side panel:

1. **Index** (once per page, again if it changes): blocks → sections → chunks → FAISS + BM25.
2. **Ask** (per question): hybrid candidates → rerank → top-5 with section labels → cited, streamed answer → trace.


Detailed file-by-file explanation is in `EXPLANATION.md`.

---

## 6. Installation / setup

**Prerequisites**
- Google Chrome
- [Ollama](https://ollama.com) installed, with the model pulled once: `ollama pull llama3.2`
- Docker (recommended) **or** Python 3.10+ for running the backend directly

**Backend, Option 1: Docker (recommended)**

```bash
cp backend/.env.example backend/.env    # defaults are already set for local Ollama
docker compose up -d --build            # first build downloads torch + embedding model (~5 min)
```

**Backend, Option 2: plain Python**

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```
The first server start downloads the embedding model (~130 MB) automatically.

**Want faster answers?** In `.env` set `LLM_PROVIDER=groq` and `GROQ_API_KEY=gsk_...` (free key at https://console.groq.com). No other changes needed.

**Extension**

1. Open `chrome://extensions`
2. Turn on **Developer mode** (top right)
3. Click **Load unpacked** and choose the `extension/` folder
4. Pin AskPage from the puzzle-piece menu

---

## 7. How to run

1. Make sure Ollama is running (`ollama serve`, or the Ollama desktop app).
2. Start the backend:
   - Docker: `docker compose up -d` (add `--build` after code changes)
   - Plain Python: `cd backend && source venv/bin/activate && export $(cat .env | xargs) && uvicorn main:app --reload --port 8000`

Check it: open http://localhost:8000/health → `{"status":"ok","provider":"ollama","model":"llama3.2",...}`

Then open any article in Chrome and click the AskPage icon.

**Run the evaluation**
```bash
cd backend
python eval/evaluate_retrieval.py   # hit@k, MRR and latency per funnel stage; no LLM needed
python eval/evaluate.py             # adds answer accuracy and citation rate; needs Ollama or Groq
```

Results are written to `eval/results_retrieval.json` and `eval/results.json`. Measured numbers are in [RESULTS.md](RESULTS.md).

---

## 8. Example usage

Open the FastAPI docs page and ask:

> **Q:** What does the `--reload` flag do?
> **A:** It restarts the server whenever you change the code. It's meant for development, not production.
> *3 passages used · retrieved in 0.04s*

> **Q:** What database does FastAPI use by default?
> **A:** I couldn't find that on this page.

Test the backend without the extension:

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"url":"test","title":"Test","text":"The Eiffel Tower is 330 metres tall and was completed in 1889."}'

curl -N -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"url":"test","question":"How tall is the Eiffel Tower?"}'
```

---

## 9. Important concepts

**RAG (Retrieval-Augmented Generation)**: Instead of hoping the LLM memorised the page, we *retrieve* the relevant passages first and put them in the prompt. The model only has to read, not recall. This is why a 3B model gives accurate answers here.

**Section-aware, sentence-aware chunking**: The extension emits the page as blocks (headings, paragraphs, lists, code, tables). `ingest.py` groups blocks under their heading path, then packs whole sentences into ~220-word chunks with a two-sentence overlap. Code and table blocks are never split. Every chunk knows its section, which is what makes citations meaningful.

**Hybrid search**: Embeddings capture meaning but blur exact tokens; BM25 is the reverse. We run both, fuse with Reciprocal Rank Fusion (`1/(60+rank)` per list, scale-free, so cosine and BM25 scores can be combined without normalising), and pass ~20 candidates on.

**Cross-encoder reranking**: A bi-encoder embeds question and chunk separately; a cross-encoder reads them *together* and outputs a relevance logit. Slower per pair (so only on the candidate pool) but much sharper. Chunks below `RERANK_MIN_SCORE` are dropped as noise, which is why unanswerable questions get few or weak sources and the model declines.

**Citations**: Sources are numbered and labelled with their section in the prompt; the system prompt requires `[n]` after each claim. The UI turns `[n]` into chips; clicking one runs `window.find` inside the page and highlights the passage with the CSS Custom Highlight API.

**Traces & metrics**: Each `/ask` becomes a trace (timings per stage, chunk ids, top rerank score, whether the model declined). `/metrics` aggregates p50/p95 latency per stage, grounded rate and error count over the last 500 requests. Every trace is also appended to `data/traces.jsonl` on a Docker volume and logged as one JSON line.

**Freshness**: The index is keyed by URL but stamped with a content hash. `/index` is idempotent: same hash → cached, different hash → re-indexed. Before every question the extension re-hashes the page and re-indexes if it changed. Staleness is event-driven, so there is no expiry timer to tune.

**Embeddings**: A model turns text into a vector (here 384 numbers). Texts with similar meaning get vectors that point in similar directions. We normalise vectors to unit length so the inner product equals cosine similarity.

**FAISS**: Facebook's library for vector search. `IndexFlatIP` does brute-force exact search, which is the right choice for hundreds of vectors. Approximate indexes (IVF, HNSW) only matter at millions of vectors.

**Query prefix for bge**: bge-style models were trained to expect a short instruction before *queries* (not documents). Adding it improves retrieval for short questions.

**Prompt grounding**: The system prompt tells the model to answer only from the provided context and to reply with a fixed sentence when it can't. This is the simplest effective guard against hallucination.

**Streaming (NDJSON)**: The backend yields one JSON object per line. The extension reads the byte stream, splits on `\n`, parses each line. First line carries the sources, then tokens, then `done`. No WebSockets required.

**CORS**: The extension's origin (`chrome-extension://...`) differs from the server's, so the browser blocks the request unless the server sends CORS headers. FastAPI's `CORSMiddleware` adds them. `ALLOWED_ORIGINS` can restrict it to the extension's exact ID.

**Rate limiting**: A dict of `client IP → recent timestamps`. Before each `/ask`, drop timestamps older than 60 s and reject if too many remain. Protects the LLM (and a Groq free-tier key, if used) from abuse. Simple, in-memory, per-process.

**Docker**: The `Dockerfile` installs CPU-only PyTorch (saves ~2 GB), pre-downloads the embedding model at build time, and runs Uvicorn on `0.0.0.0` so it's reachable from outside the container. `docker-compose.yml` wires in the `.env` file and auto-restarts on crash.

**Provider switching**: `llm.py` reads `LLM_PROVIDER` once and dispatches in `stream_chat`. Both Groq and Ollama accept the same `[{role, content}]` message list, so `build_messages` is shared and only the API call differs.

**Chrome Manifest V3**: `chrome.scripting.executeScript` runs a function *inside* the web page to read its DOM; the side panel itself can't touch the page directly. `activeTab` grants that access only for the tab the user is on.

**Evaluation**: Each test question carries an `evidence` substring. Retrieval is scored as hit@k and MRR (did the evidence chunk come back, at what rank?) in four configurations, dense only, sparse only, RRF fusion, and fusion + cross-encoder rerank, so the gain from each stage is visible and no row silently includes a stage it does not name. Measured results are in [RESULTS.md](RESULTS.md): on 40 questions across 4 real pages, the full funnel reaches hit@1 0.844 / MRR 0.917, with the evidence chunk in the top 5 for 100% of answerable questions. Answers are graded by LLM-as-judge; unanswerable questions count as correct only if the model declined.

---

## 10. Future improvements

- Persist indexes to disk so they survive server restarts
- Multi-turn chat with conversation history in the prompt
- Support PDFs opened in the browser
- Mozilla Readability as an additional extraction strategy for very messy pages
- Retrieval-only "confidence" gate: skip the LLM when the top rerank score is low
- Export traces via OpenTelemetry; Redis-backed rate limiting for multi-replica deployments
- Larger eval set (100+ questions) across messier page types, with comparison tables: MiniLM vs bge, 1B vs 3B, with/without reranker
