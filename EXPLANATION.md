# AskPage: step-by-step explanation

Read this top to bottom once and you should be able to follow every part of the codebase.

---

## Part 0: the big picture

Two programs cooperate:

- The **extension** lives inside Chrome. It can see the web page but cannot run Python or ML models.
- The **backend** is a Python server on `localhost:8000`. It can run ML models but cannot see the web page.

So the extension *sends* the page text to the backend once, then sends *questions*. The backend answers. That's the whole architecture.

---

## Part 1a: `backend/ingest.py`: smart ingestion

*Given a raw page, produce chunks worth searching.* Four steps, each a function:

### 1a.1 `clean_text`
NFKC-normalises unicode (so "ﬁ" ligatures and full-width digits compare equal), strips zero-width characters, collapses whitespace. `is_boilerplate` drops lines like "Accept all cookies" or "Skip to content" using a small regex list.

### 1a.2 `build_sections`
The extension sends **blocks**: `{type: heading|paragraph|list|code|table|quote, level, text}` in reading order. We walk them keeping a *heading stack*: an `h2` pops any open `h2`/`h3`, an `h3` nests under the current `h2`. Every non-heading block is attached to the section that's open at that moment, so each section knows its full heading path, e.g. `["Ops Runbook", "Error codes"]`.

If there are no blocks (old extension, or plain text from the eval set), `blocks_from_text` fakes one paragraph block per line.

### 1a.3 `chunk_section`
Each section's text is split into sentences with a regex (`.`/`!`/`?` followed by whitespace and a capital letter/digit, no NLP dependency). Sentences are packed into a buffer until adding the next one would pass `CHUNK_SIZE_WORDS` (220); then the buffer becomes a chunk and its last two sentences are carried into the next buffer as overlap. Code and table blocks are treated as single indivisible units so they're never cut mid-way. A trailing chunk under 15 words is merged into its predecessor.

Every chunk gets `section`, `heading_path` and `block_types` metadata. That metadata is what the prompt and the UI use for citations.

### 1a.4 `ingest` + dedupe
Runs the pipeline and drops any chunk whose lower-cased, punctuation-stripped text has already been seen. Repeated share bars and cookie notices disappear here. Chunks get ids `c0, c1, …` in page order.

---

## Part 1b: `backend/rag.py`: hybrid retrieval and reranking

*Given a page and a question, which chunks are relevant?*

### 1b.1 Three models/indexes, one page store
```python
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker        = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
page_store[url_hash] = {chunks, index (FAISS), bm25, content_hash, indexed_at, title}
```
Both models load once at import. The store is keyed by `sha256(url)`.

### 1b.2 `index_page`
1. `ingest.ingest(blocks, text)` → chunks.
2. `content_hash` = sha256 of all chunk text. If the page is already stored with the same hash → return `cached: True` and refresh its timestamp. Different hash → fall through and re-index (`refreshed: True`). This is the freshness mechanism.
3. Embed chunks (normalised → inner product = cosine) into `IndexFlatIP`.
4. Build `BM25Okapi` over `tokenize(chunk)`, lowercase word tokens minus a tiny stopword list. Tokens keep `-`, `_`, `.`, `/` so `--reload` and `data/traces.jsonl` survive as single terms.

### 1b.3 `retrieve_chunks`: the funnel
```
vector top-12 ─┐
               ├─ RRF ─→ candidates ─→ cross-encoder ─→ threshold ─→ top-5
BM25   top-12 ─┘
```
- **RRF** (`_reciprocal_rank_fusion`): each list gives item `1/(60 + rank)`. Ranks, not raw scores, so a cosine of 0.7 and a BM25 of 4.2 combine without any normalisation.
- **Rerank**: `reranker.predict([(question, chunk_text), …])` returns one logit per pair. Positive ≈ relevant, below −5 ≈ irrelevant. We sort by it, drop anything under `RERANK_MIN_SCORE` (−6) while always keeping at least 2, and cut to `TOP_K`.
- `mode="vector"|"bm25"|"hybrid"` exists so `evaluate.py` can measure each stage on its own.

Returned chunks carry `vector_score`, `bm25_score`, `rerank_score` and their ingestion metadata; `timings` has `retrieval` and `rerank` separately for the trace.

### 1b.4 TTL
`_get_page` raises `KeyError` if the page was never indexed. `main.py` turns that into a 404; the extension reacts by re-indexing and retrying once.

---

## Part 2: `backend/llm.py`: generation

This file answers: *given relevant chunks and a question, what's the answer?*

### 2.0 Provider switch

At import time we read three environment variables:

```python
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
MODEL_NAME   = os.getenv("LLM_MODEL", DEFAULT_MODELS[LLM_PROVIDER])
groq_client  = groq.Groq() if LLM_PROVIDER == "groq" else None
```

`groq.Groq()` reads `GROQ_API_KEY` from the environment by itself. We only build the client when Groq is selected, so Ollama users never need a key.

Both providers take the same `[{role, content}]` message list, so `build_messages` is shared. Only the network call differs: `_stream_groq` uses the OpenAI-style `chat.completions.create(..., stream=True)` and reads `chunk.choices[0].delta.content`; `_stream_ollama` uses `ollama.chat(..., stream=True)` and reads `part["message"]["content"]`. `stream_chat` picks one. Everything above that line is provider-agnostic.

Each provider maps its own exceptions (`groq.AuthenticationError`, `ollama.ResponseError`, …) to a plain `RuntimeError` with a human message, so `main.py` only has to catch one type.

### 2.1 The system prompt
Four rules: answer only from the sources; decline with an exact sentence if they don't contain it; **cite `[n]` after each claim**; be concise. The fixed decline sentence is what `observability.py` and `evaluate.py` look for to detect a refusal.

### 2.2 `build_messages`
Each chunk becomes `[n] (section: Heading > Subheading)\n<text>`. Numbering the sources is what makes `[n]` citations possible; the section label lets the model say *where* on the page, and the UI shows the same label on the source card.

### 2.3 `stream_answer` / `generate_answer`

`stream_answer` is a Python **generator** (uses `yield from`). It builds the messages and delegates to `stream_chat`. `generate_answer` is the same thing joined into one string, used by the eval script, which doesn't need streaming. `complete_chat` is the generic non-streaming helper the judge in `evaluate.py` uses.

---

## Part 3: `backend/main.py`: the API

### 3.1 CORS
The extension's origin is `chrome-extension://<id>`. The server is `http://localhost:8000`. Different origins → browser blocks the request unless the server says it's OK via CORS headers. `ALLOWED_ORIGINS` controls the list; `*` (default) is fine locally.

### 3.2 Rate limiting

### 3.3 Rate limiting
`recent_asks[client_ip]` is a list of timestamps. `check_rate_limit` drops entries older than 60 s and raises 429 if `MAX_ASKS_PER_MINUTE` remain.

### 3.4 Request models
`IndexRequest` accepts either `blocks: list[Block]` (preferred) or `text`. `FeedbackRequest.rating` is validated against `^(up|down)$` by Pydantic, a bad value is a 422 before our code runs.

### 3.5 `/index`
Caps the total block text at 200k characters, calls `rag.index_page(...)`, returns `{cached, refreshed, num_chunks, num_sections, content_hash}`.

### 3.6 `/ask`
1. Create a `Trace` (see Part 3b).
2. Retrieval + rerank run **before** streaming so a missing/expired page can still be a proper 404.
3. `event_stream()` yields NDJSON: a `sources` line (with `trace_id`, chunks, timings, mode), `token` lines, then `done` with the `trace_id` again. The first token's arrival time is recorded as `first_token`.
4. Any exception is caught and sent as an `error` line, if it propagated, uvicorn would cut the connection and Chrome would show only "network error".
5. `trace.finish(answer)` records totals and writes the trace.

### 3.7 `/metrics`
Thin wrappers over `observability.py`.

---

## Part 3b: `backend/observability.py`: traces and metrics

**`Trace`** is a small notebook for one `/ask`: question, url hash, provider/model, per-stage timings, retrieval summary (candidate count, chunk ids, top rerank score), answer length, `grounded` (False if the decline sentence appeared), error. `finish()` pushes it into a 500-entry ring buffer, appends it to `data/traces.jsonl`, and logs it as one JSON line, so `docker compose logs | grep '"event": "ask"'` is a usable audit trail.

**`metrics_summary`** computes over the ring buffer: p50/p95/mean for retrieval, rerank, first-token and total latency; grounded rate; distribution of top rerank scores (a falling p50 here is an early warning that retrieval quality is slipping).

---

## Part 4: `backend/eval/evaluate.py`: measuring quality

Each test question has `expected_answer` and an `evidence` string that must occur in the chunk containing the answer (`null` for questions that aren't on the page).

For every question:
1. Retrieve in four configurations, `dense`, `sparse`, `hybrid`, `hybrid+rerank`, and record the rank of the first chunk containing `evidence`. Aggregated as **hit@1/3/5** and **MRR** per configuration. Each row names whether the cross-encoder ran, so no baseline silently includes a stage it doesn't claim.
2. Generate an answer with the hybrid chunks and grade it with the LLM judge (YES/NO).
3. For `evidence: null` questions, the answer is correct only if the model **declined**; answering anyway counts toward the **false-answer rate**.
4. Record whether the answer contained a `[n]` citation (**citation rate**) and per-stage latency (p50/p95).

Everything is saved to `results.json`. Run it inside the container: `docker compose exec backend python eval/evaluate.py`.

---

## Part 4b: Docker

**`backend/Dockerfile`**: four steps: install CPU-only torch (the default one drags in ~2 GB of CUDA), install `requirements.txt`, pre-download the embedding model *and* the cross-encoder reranker so startup is instant, copy the code. `CMD` runs Uvicorn on `0.0.0.0`, inside a container `localhost` would only be reachable from inside, `0.0.0.0` means "all interfaces".

**`docker-compose.yml`**: builds that image, maps port 8000, loads `backend/.env`, points `OLLAMA_HOST` at `host.docker.internal` (the container's name for your machine, where Ollama runs), mounts `backend/data` as a volume so traces survive rebuilds, and `restart: unless-stopped` brings it back after a crash or reboot.

**`.env` / `.env.example`**: secrets and config live here, never in code. `.env` is git-ignored and docker-ignored; `.env.example` is the committed template.

---

## Part 5: `extension/manifest.json`

| Key | Meaning |
|-----|---------|
| `manifest_version: 3` | Current Chrome extension format |
| `permissions.sidePanel` | Allowed to show a side panel |
| `permissions.activeTab` | Allowed to read the tab the user is on, after they click the icon |
| `permissions.scripting` | Allowed to inject a script into that tab |
| `permissions.storage` | Remember the backend URL in `chrome.storage.local` |
| `host_permissions` | Allowed to `fetch` localhost:8000, and to read the text of any http/https page you open |
| `action` | The toolbar icon |
| `side_panel.default_path` | Which HTML to show in the panel |
| `background.service_worker` | Script that runs in the background |

---

## Part 6: `extension/background.js`

One call: `setPanelBehavior({ openPanelOnActionClick: true })`. Without it, clicking the icon does nothing.

---

## Part 7: `extension/sidepanel.html` + `.css`

Three regions top to bottom: header (title + status), chat (messages), composer (textarea + button). The textarea and button start `disabled` and are enabled once indexing finishes.

CSS is deliberately plain. One thing worth knowing: `.message.streaming::after` draws a blinking cursor while tokens are arriving, removed when the stream ends.

---

## Part 8: `extension/sidepanel.js`: the glue

### 8.1 `extractPageContent`
Passed to `chrome.scripting.executeScript({ func })`. Chrome serialises it, injects it into the web page, runs it there, and returns its result. That's why it uses the page's `document` and can't reference anything else in this file.

It clones the body, strips noise (`nav`, `footer`, `script`, `[aria-hidden]`, …), prefers `<article>`/`<main>`, then **walks the DOM** emitting blocks: `h1–h6` → `heading` with a level; `p`, `li`, `pre`, `blockquote`, `td`, `dt/dd` → typed leaf blocks; unknown elements with only inline content → `paragraph`; text sitting loose inside a `div` → `paragraph` if it's more than a few words. A leaf that contains other leaves (`<li><p>…`) is recursed into instead of emitted, so nothing is sent twice. The flat `text` is still included as a fallback and for hashing.

### 8.2 `readTab` / `hashText`
`readTab` refuses non-`http` tabs and runs the extractor. `hashText` SHA-256s the page text in the browser with `crypto.subtle`, cheap, and it lets the panel notice a changed page without a round trip.

### 8.3 Settings + `apiPost`
`settings` (`backendUrl`) loads from `chrome.storage.local` on start and is edited via the ⚙ panel. `apiPost` turns FastAPI's `{detail}` errors into `Error` objects that also carry `.status`, so callers can react to 404 by re-indexing.

### 8.4 `prepareTab` / `ensureFresh`
`prepareTab` = extract → `/index` → store the content hash → "Ready · N chunks · M sections". `ensureFresh` runs before every question: re-extract, re-hash, and if the hash differs, `/index` again (the backend answers `refreshed: true`). If `/ask` still returns 404 (index expired or backend restarted), `handleAsk` re-indexes once and retries.

### 8.5 `askQuestion`: reading the stream
`reader.read()` gives raw bytes in arbitrary sizes; we decode, split on `\n`, keep the incomplete tail in `leftover`, and parse each full line. `sources` fills the collapsible card (section label, rerank score, snippet) and remembers `trace_id`; `token` appends text; `done` confirms the trace id. When the stream ends, `renderCitations` runs.

### 8.6 `renderCitations` / `highlightOnPage`
The answer text is split on `[n]` / `[2][3]` patterns. Each number becomes a `<button class="cite">` whose tooltip is the source's heading path. Clicking it (or a source card) injects `highlightSnippetInPage` into the tab: it tries `window.find` with the chunk's first 40, 25, 15, then 8 words (layout can split a chunk across elements), wraps the match in a `Highlight` registered under `CSS.highlights` so it's painted yellow via `::highlight(askpage)`, and scrolls it into view.

### 8.7 `addFeedbackRow`

---

## Part 9: full data flow, end to end

1. User clicks the AskPage icon → `background.js` opens the panel → `sidepanel.js` loads settings and runs `switchToTab`.
2. `extractPageContent` runs inside the page → `{url, title, blocks[], text}`.
3. `POST /index` → `ingest.ingest` (clean, sections, sentence chunks, dedupe) → `rag.index_page` stores FAISS + BM25 under `sha256(url)` with a content hash.
4. Panel shows "Ready · 42 chunks · 9 sections". Input enabled.
5. User asks → `ensureFresh` re-hashes the page (re-indexes if changed) → `POST /ask`.
6. `rag.retrieve_chunks`: vector top-12 ∪ BM25 top-12 → RRF → cross-encoder → threshold → top-5.
7. `llm.stream_answer` builds the numbered, section-labelled prompt; the model streams a cited answer.
8. Server streams `sources` (with `trace_id`) → `token`s → `done`; `Trace.finish` writes timings and chunk ids to `traces.jsonl` and the log.
9. Panel renders `[n]` chips; clicking one highlights the passage on the page.
10. `/metrics` shows p50/p95 latency per stage and grounded rate; `evaluate.py` reports hit@k/MRR per funnel stage and judged accuracy.

---

## Part 10: the concepts, one line each

- **RAG**: retrieve relevant text first, then let the LLM read it.
- **Embedding**: text → vector; similar meaning → similar direction.
- **Cosine similarity via inner product**: works because vectors are normalised.
- **FAISS IndexFlatIP**: exact search; right for small collections.
- **Section-aware chunking**: chunks are cut at sentence ends under their heading path; code/tables stay whole.
- **Content hash**: same page text → cached; different → re-indexed. Freshness without a scheduler.
- **BM25**: keyword scoring; catches exact tokens embeddings blur.
- **RRF**: rank-based fusion; combines lists with incomparable scores.
- **Cross-encoder**: scores (question, passage) jointly; used only on the candidate pool.
- **Relevance threshold**: drop weak reranker scores so the model sees less noise.
- **Citations `[n]`**: numbered sources in the prompt; chips in the UI; `window.find` + CSS Highlight on the page.
- **Trace**: one record per request with per-stage timings; JSONL on disk + `/metrics`.
- **Owner namespace**: `(api key or IP, url)` keys the store; nobody reads another's pages.
- **Grounded prompt**: "answer only from context, else say not found."
- **Generator + StreamingResponse**: yield strings, browser receives them live.
- **NDJSON**: one JSON per line; trivial to parse incrementally.
- **CORS**: server permission for cross-origin browser requests; locked to the extension ID in production.
- **Rate limiting**: per-IP timestamp list; cheap protection for a public free-tier API.
- **Provider switch**: one env var chooses Groq or Ollama; message format is shared.
- **Docker**: reproducible image with the model baked in; `compose up` deploys it.
- **executeScript**: run your function inside the web page from the extension.
- **LLM-as-judge**: grade free-text answers with the model itself.
- **hit@k / MRR**: did the evidence chunk come back, and how high? Measured per retrieval mode.
