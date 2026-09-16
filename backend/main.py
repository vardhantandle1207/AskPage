"""
main.py — the FastAPI server. This is the only file the extension talks to.

Routes:
    GET  /health    -> quick check that the server is alive
    POST /index     -> receive page blocks/text, ingest + index (re-indexes if changed)
    POST /ask       -> retrieve (hybrid + rerank), stream a cited answer
    GET  /metrics   -> latency percentiles, grounded rate, error count

Run locally:   uvicorn main:app --reload --port 8000
Run in Docker: see Dockerfile / docker-compose.yml
"""

import json
import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import llm
import observability
import rag

logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(title="AskPage API")

# ---------------------------------------------------------------------------
# CORS — the extension's chrome-extension:// origin differs from ours.
# ---------------------------------------------------------------------------
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rate limiting: per client IP, in memory. Only matters for a deployed demo;
# a dict of recent timestamps is enough at this scale.
# ---------------------------------------------------------------------------
MAX_ASKS_PER_MINUTE = int(os.getenv("MAX_ASKS_PER_MINUTE", "10"))
recent_asks: dict[str, list[float]] = {}


def check_rate_limit(client: str):
    now = time.time()
    timestamps = [t for t in recent_asks.get(client, []) if now - t < 60]
    if len(timestamps) >= MAX_ASKS_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Too many questions. Limit is {MAX_ASKS_PER_MINUTE} per minute.",
        )
    timestamps.append(now)
    recent_asks[client] = timestamps


MAX_TEXT_CHARACTERS = 200_000   # ~40k words; keeps indexing fast


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class Block(BaseModel):
    type: str = "paragraph"        # heading | paragraph | list | code | table | quote
    level: int | None = None       # headings only, 1-6
    text: str


class IndexRequest(BaseModel):
    url: str
    title: str = ""
    text: str | None = None        # flat fallback
    blocks: list[Block] | None = None


class AskRequest(BaseModel):
    url: str
    question: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": llm.LLM_PROVIDER,
        "model": llm.MODEL_NAME,
        "reranker": rag.RERANKER_MODEL_NAME if rag.reranker else None,
        "pages_indexed": rag.pages_indexed(),
    }


@app.post("/index")
def index_page(request: IndexRequest):
    """Ingest and index a page. Re-indexes if its content changed."""
    blocks = None
    if request.blocks:
        blocks, total = [], 0
        for block in request.blocks:
            total += len(block.text)
            if total > MAX_TEXT_CHARACTERS:
                break
            blocks.append(block.model_dump())
    text = (request.text or "").strip()[:MAX_TEXT_CHARACTERS]

    if not blocks and not text:
        raise HTTPException(status_code=400, detail="Page text is empty.")
    try:
        return rag.index_page(request.url, request.title, text=text, blocks=blocks)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/ask")
def ask_question(request: AskRequest, http_request: Request):
    """
    Stream an answer as NDJSON:
        {"type": "sources", "trace_id": ..., "chunks": [...], "timings": {...}}
        {"type": "token", "text": "The"} ...
        {"type": "done", "trace_id": ...}
    or  {"type": "error", "message": "..."}
    """
    client = http_request.client.host if http_request.client else "unknown"
    check_rate_limit(client)
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    trace = observability.Trace(rag.get_url_hash(request.url), question,
                                llm.LLM_PROVIDER, llm.MODEL_NAME)

    # Step 1: retrieval + rerank (fast, before streaming starts)
    try:
        retrieval = rag.retrieve_chunks(request.url, question)
        title = rag.page_title(request.url)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=error.args[0])
    trace.set_retrieval(retrieval)

    # Step 2: generation, streamed
    def event_stream():
        yield json.dumps({
            "type": "sources",
            "trace_id": trace.id,
            "chunks": retrieval["chunks"],
            "timings": retrieval["timings"],
            "mode": retrieval["mode"],
            "retrieval_time_seconds": retrieval["retrieval_time_seconds"],
        }) + "\n"

        answer_parts: list[str] = []
        t0 = time.time()
        try:
            for token in llm.stream_answer(question, retrieval["chunks"], title):
                if not answer_parts:
                    trace.timing("first_token", time.time() - t0)
                answer_parts.append(token)
                yield json.dumps({"type": "token", "text": token}) + "\n"
        except RuntimeError as error:
            trace.finish("".join(answer_parts), error=str(error))
            yield json.dumps({"type": "error", "message": str(error)}) + "\n"
            return
        except Exception as error:  # noqa: BLE001
            # If this propagated, uvicorn would drop the connection mid-stream
            # and the browser would just show "network error".
            logging.exception("LLM generation failed")
            message = f"Generation failed: {type(error).__name__}: {error}"
            trace.finish("".join(answer_parts), error=message)
            yield json.dumps({"type": "error", "message": message}) + "\n"
            return

        trace.timing("generation", time.time() - t0)
        trace.finish("".join(answer_parts))
        yield json.dumps({"type": "done", "trace_id": trace.id}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/metrics")
def metrics():
    return observability.metrics_summary(rag.pages_indexed())
