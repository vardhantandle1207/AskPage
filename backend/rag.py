"""
rag.py — the retrieval part of the RAG pipeline: hybrid search + reranking.

For every indexed page we keep three things:
    1. the chunks themselves (from ingest.py, with section metadata),
    2. a FAISS index of their embeddings      -> semantic search
    3. a BM25 index of their words            -> lexical (keyword) search

Answering a question is a three-stage funnel:

    question ──► vector top-N ──┐
                                ├─► fuse (RRF) ──► cross-encoder rerank ──► top-K
    question ──► BM25  top-N ───┘

Why both? Vectors understand meaning ("cost" ≈ "budget") but miss exact
tokens (error codes, names, numbers). BM25 is the opposite. Fusing them
gives a candidate pool that rarely misses. The cross-encoder then reads
question + chunk *together* and produces a much sharper relevance score
than either retriever, so noise drops out and the best passage floats up.

Freshness: each page stores a hash of its content. The extension re-sends the
page on every visit; if the hash matches we reuse the existing index, and if
the page changed we rebuild it. That makes staleness event-driven rather than
time-based.
"""

import hashlib
import os
import re
import time

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import ingest

# ---------------------------------------------------------------------------
# Settings (env vars let you experiment without editing code)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Cross-encoder trained on MS MARCO passage ranking. ~80 MB, CPU-friendly.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATES_PER_RETRIEVER = 12   # top-N from each of vector and BM25
TOP_K = int(os.getenv("TOP_K", "5"))               # chunks handed to the LLM
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") != "0"
# Cross-encoder scores are logits: > 0 clearly relevant, < -5 clearly not.
# Chunks below this are dropped as noise (we always keep at least MIN_KEEP).
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "-6"))
MIN_KEEP = 2
RRF_K = 60                      # standard constant for reciprocal rank fusion

MAX_STORED_PAGES = int(os.getenv("MAX_STORED_PAGES", "200"))

# ---------------------------------------------------------------------------
# Models + global state
# ---------------------------------------------------------------------------

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME) if RERANK_ENABLED else None

# page_store[url_hash] = {
#   "title", "url", "chunks", "index" (faiss), "bm25", "content_hash", "indexed_at"
# }
page_store: dict[str, dict] = {}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "it", "this", "that", "with", "as", "by", "at", "from",
    "what", "which", "who", "how", "when", "where", "why", "does", "do", "did",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def content_hash(chunks: list[dict]) -> str:
    """Fingerprint of the page content, so we can tell when it changed."""
    joined = "\n".join(c["text"] for c in chunks)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def search_text(chunk: dict) -> str:
    """
    What the retrievers actually see: the heading path prepended to the
    chunk. "Rotating the reranker > To disable the cross-encoder..." lets a
    question about "the reranker" match a chunk that never uses the word.
    The plain `text` is still what the LLM and the UI get.
    """
    path = " > ".join(chunk.get("heading_path") or [])
    return f"{path}\n{chunk['text']}" if path else chunk["text"]


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens minus stopwords — what BM25 sees."""
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-_./]*", text.lower())
            if t not in _STOPWORDS]


def embed_texts(texts: list[str]) -> np.ndarray:
    vectors = embedding_model.encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32")


def _get_page(url: str) -> dict:
    page = page_store.get(get_url_hash(url))
    if page is None:
        raise KeyError("Page has not been indexed yet. Call /index first.")
    return page


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_page(url: str, title: str, text: str | None = None,
               blocks: list[dict] | None = None) -> dict:
    """
    Ingest + embed + index one page.
    Re-indexes silently if the page content changed since last time.
    """
    key = get_url_hash(url)
    start_time = time.time()

    chunks = ingest.ingest(blocks, text)
    if not chunks:
        raise ValueError("The page has no readable text to index.")
    fingerprint = content_hash(chunks)

    existing = page_store.get(key)
    if existing and existing["content_hash"] == fingerprint:
        existing["indexed_at"] = time.time()      # content unchanged: refresh TTL
        return {"cached": True, "refreshed": False, "num_chunks": len(existing["chunks"]),
                "content_hash": fingerprint}

    searchable = [search_text(c) for c in chunks]
    vectors = embed_texts(searchable)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    bm25 = BM25Okapi([tokenize(t) for t in searchable])

    if existing is None and len(page_store) >= MAX_STORED_PAGES:
        del page_store[next(iter(page_store))]     # evict the oldest page

    page_store[key] = {
        "title": title,
        "url": url,
        "chunks": chunks,
        "index": index,
        "bm25": bm25,
        "content_hash": fingerprint,
        "indexed_at": time.time(),
    }
    return {
        "cached": False,
        "refreshed": existing is not None,
        "num_chunks": len(chunks),
        "num_sections": len({tuple(c["heading_path"]) for c in chunks}),
        "content_hash": fingerprint,
        "index_time_seconds": round(time.time() - start_time, 2),
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _vector_search(page: dict, question: str, n: int) -> list[tuple[int, float]]:
    q = embed_texts([QUERY_PREFIX + question])
    scores, positions = page["index"].search(q, min(n, len(page["chunks"])))
    return [(int(p), float(s)) for p, s in zip(positions[0], scores[0]) if p >= 0]


def _bm25_search(page: dict, question: str, n: int) -> list[tuple[int, float]]:
    tokens = tokenize(question)
    if not tokens:
        return []
    scores = page["bm25"].get_scores(tokens)
    order = np.argsort(scores)[::-1][:n]
    # A chunk with no matching token scores exactly 0. (On a page with only
    # one or two chunks BM25's IDF can go negative, so don't require > 0.)
    return [(int(p), float(scores[p])) for p in order if scores[p] != 0]


def _reciprocal_rank_fusion(*ranked_lists: list[tuple[int, float]]) -> dict[int, float]:
    """
    RRF: each list votes 1/(k + rank) for its items. Scale-free, so vector
    cosine scores and BM25 scores can be combined without normalising.
    """
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (position, _) in enumerate(ranked):
            fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank + 1)
    return fused


def retrieve_chunks(url: str, question: str, top_k: int = TOP_K,
                    mode: str = "hybrid", rerank: bool | None = None) -> dict:
    """
    Find the best chunks for `question`.

    mode:   "hybrid" (default) | "vector" | "bm25"
    rerank: None (default) = use the global RERANK_ENABLED setting.
            True/False overrides it for this call.

    The two knobs are separate on purpose: the evaluation script turns the
    cross-encoder off to measure what fusion alone buys, then on to measure
    what reranking adds on top. Without that separation a "vector" number is
    really "vector + rerank" and the funnel can't be attributed.

    Returns {"chunks": [...], "num_candidates": int, "mode": str,
             "timings": {"retrieval": s, "rerank": s}}.
    """
    page = _get_page(url)
    chunks = page["chunks"]
    timings = {}

    t0 = time.time()
    vector_hits = _vector_search(page, question, CANDIDATES_PER_RETRIEVER) if mode != "bm25" else []
    bm25_hits = _bm25_search(page, question, CANDIDATES_PER_RETRIEVER) if mode != "vector" else []
    fused = _reciprocal_rank_fusion(vector_hits, bm25_hits)
    candidates = sorted(fused, key=fused.get, reverse=True)
    timings["retrieval"] = round(time.time() - t0, 4)

    vector_score = dict(vector_hits)
    bm25_score = dict(bm25_hits)

    # Rerank the candidate pool with the cross-encoder
    t1 = time.time()
    use_reranker = RERANK_ENABLED if rerank is None else rerank
    if reranker is not None and use_reranker and candidates:
        pairs = [(question, search_text(chunks[p])) for p in candidates]
        rerank_scores = reranker.predict(pairs).tolist()
        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        kept = [(p, s) for i, (p, s) in enumerate(ranked)
                if s >= RERANK_MIN_SCORE or i < MIN_KEEP][:top_k]
    else:
        kept = [(p, None) for p in candidates[:top_k]]
    timings["rerank"] = round(time.time() - t1, 4)

    results = []
    for position, rerank_score in kept:
        chunk = chunks[position]
        results.append({
            **chunk,
            "vector_score": round(vector_score.get(position, 0.0), 3),
            "bm25_score": round(bm25_score.get(position, 0.0), 3),
            "rerank_score": round(rerank_score, 3) if rerank_score is not None else None,
            "score": round(rerank_score if rerank_score is not None else fused[position], 3),
        })

    return {
        "chunks": results,
        "num_candidates": len(candidates),
        "mode": mode + ("+rerank" if (reranker is not None and use_reranker) else ""),
        "timings": timings,
        # kept for the old UI/evaluate.py field name
        "retrieval_time_seconds": round(timings["retrieval"] + timings["rerank"], 4),
    }


def page_title(url: str) -> str:
    return _get_page(url)["title"]


def pages_indexed() -> int:
    return len(page_store)
