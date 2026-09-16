"""
observability.py — know what the pipeline is doing and how well.

Every /ask produces one trace: what was asked, how long each stage took,
which chunks were used, and whether the answer was grounded in the page. The
last few hundred traces stay in memory for /metrics; every trace is also
written as one JSON line to data/traces.jsonl so nothing is lost on restart.

Structured logging: each trace is also logged as a single JSON line, so
`docker compose logs` can be grepped or shipped anywhere.
"""

import json
import logging
import os
import statistics
import threading
import time
import uuid
from collections import deque

DATA_DIR = os.getenv("ASKPAGE_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
TRACES_PATH = os.path.join(DATA_DIR, "traces.jsonl")
MAX_TRACES_IN_MEMORY = 500

logger = logging.getLogger("askpage")

_lock = threading.Lock()
_traces: deque = deque(maxlen=MAX_TRACES_IN_MEMORY)
_started_at = time.time()


def _append_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

class Trace:
    """
    One /ask request, start to finish. Use it like a notebook:

        trace = Trace(url_hash, question)
        trace.timing("retrieval", 0.032)
        ...
        trace.finish(answer_text)
    """

    def __init__(self, url_hash: str, question: str, provider: str, model: str):
        self.data = {
            "trace_id": uuid.uuid4().hex[:12],
            "timestamp": time.time(),
            "url_hash": url_hash,
            "question": question,
            "provider": provider,
            "model": model,
            "timings": {},        # stage -> seconds
            "retrieval": {},      # counts + scores from rag.retrieve_chunks
            "answer_chars": 0,
            "grounded": None,     # False if the model said it couldn't find it
            "error": None,
        }
        self._t0 = time.time()

    @property
    def id(self) -> str:
        return self.data["trace_id"]

    def timing(self, stage: str, seconds: float):
        self.data["timings"][stage] = round(seconds, 4)

    def set_retrieval(self, retrieval: dict):
        self.data["retrieval"] = {
            "candidates": retrieval.get("num_candidates"),
            "returned": len(retrieval.get("chunks", [])),
            "top_rerank_score": (retrieval["chunks"][0].get("rerank_score")
                                 if retrieval.get("chunks") else None),
            "chunk_ids": [c.get("id") for c in retrieval.get("chunks", [])],
            "mode": retrieval.get("mode"),
        }
        for stage, seconds in retrieval.get("timings", {}).items():
            self.timing(stage, seconds)

    def finish(self, answer: str, error: str | None = None):
        self.timing("total", time.time() - self._t0)
        self.data["answer_chars"] = len(answer)
        self.data["grounded"] = "couldn't find that on this page" not in answer.lower()
        self.data["error"] = error
        with _lock:
            _traces.append(self.data)
        _append_jsonl(TRACES_PATH, self.data)
        logger.info(json.dumps({"event": "ask", **self.data}, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "mean": None}
    ordered = sorted(values)
    def pct(p):
        index = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
        return round(ordered[index], 4)
    return {"p50": pct(0.5), "p95": pct(0.95), "mean": round(statistics.fmean(ordered), 4)}


def metrics_summary(pages_indexed: int) -> dict:
    """The numbers you'd put on a dashboard, computed over recent traces."""
    with _lock:
        traces = list(_traces)

    def stage(name):
        return _percentiles([t["timings"][name] for t in traces if name in t["timings"]])

    grounded = [t for t in traces if t["grounded"] is not None]

    return {
        "uptime_seconds": round(time.time() - _started_at),
        "pages_indexed": pages_indexed,
        "asks": len(traces),
        "errors": sum(1 for t in traces if t["error"]),
        "latency_seconds": {
            "retrieval": stage("retrieval"),
            "rerank": stage("rerank"),
            "first_token": stage("first_token"),
            "total": stage("total"),
        },
        "grounded_rate": (round(sum(1 for t in grounded if t["grounded"]) / len(grounded), 3)
                          if grounded else None),
        "top_rerank_score": _percentiles([
            t["retrieval"]["top_rerank_score"] for t in traces
            if t["retrieval"].get("top_rerank_score") is not None
        ]),
    }
