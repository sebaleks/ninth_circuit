"""FastAPI app for the asylum-case RAG system.

Endpoints:
  POST /chat    — question → answer with citations
  POST /search  — query    → top-k similar cases (no LLM)
  GET  /health  — liveness + index stats
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from rag_api import ce_verify, generation, guardrails, nvidia_client, retrieval, timing
from rag_api.models import (
    ChatRequest,
    ChatResponse,
    Citation,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    retrieval.load()
    yield


app = FastAPI(
    title="Ninth Circuit asylum-case RAG API",
    description="Retrieval-augmented Q&A over Ninth Circuit asylum opinions.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Vercel frontend + local dev
_DEFAULT_ORIGINS = [
    "https://asylum-viewer.vercel.app",
    "http://localhost:3000",
]
_EXTRA = os.environ.get("CORS_EXTRA_ORIGINS", "")
_ORIGINS = _DEFAULT_ORIGINS + [o.strip() for o in _EXTRA.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ORIGINS,
    allow_origin_regex=r"https://asylum-viewer.*\.vercel\.app",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        n_chunks=retrieval.n_chunks(),
        embedding_dim=retrieval.embedding_dim(),
        vector_store=retrieval.vector_store_name(),
        embedder=retrieval.embedder_name(),
        use_reranker=retrieval.USE_RERANKER,
        fusion_method=retrieval.FUSION_METHOD,
        bm25_backend=retrieval.BM25_BACKEND,
        ce_loaded=ce_verify.is_loaded(),
        embed_model=nvidia_client.EMBED_MODEL,
        rerank_model=nvidia_client.RERANK_MODEL,
        gen_model=nvidia_client.GEN_MODEL,
        build_sha=os.environ.get("RENDER_GIT_COMMIT", "dev")[:7],
    )


def _finalize(t: timing.Timings, t0: float, endpoint: str) -> dict:
    """Build the per-stage report, log it as a JSON line (always), and return it.

    Logging is unconditional (every request); whether the report is surfaced in
    the response body is the caller's choice via ?include_timings.
    """
    report = timing.build_report(t, (time.perf_counter() - t0) * 1000.0)
    timing.log_report(endpoint, report)
    return report


@app.post("/search", response_model=SearchResponse, response_model_exclude_none=True)
def search(req: SearchRequest, include_timings: bool = False) -> SearchResponse:
    t = timing.start()
    t0 = time.perf_counter()
    try:
        try:
            hits = retrieval.search_with_rerank(req.query, fetch_k=max(20, req.k * 2), return_k=req.k)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e

        # Apply the same dense-score refusal threshold as /chat so out-of-corpus
        # queries don't return noise. The dense score is more uniformly
        # calibrated than the rerank sigmoid.
        if guardrails.should_refuse([h.get("dense_score", h["score"]) for h in hits]):
            latency_ms = int((time.perf_counter() - t0) * 1000)
            report = _finalize(t, t0, "search")
            return SearchResponse(hits=[], latency_ms=latency_ms, refused=True,
                                  timings=report if include_timings else None)

        # The cross-encoder is intentionally OUT of this default path now: it's opt-in via
        # /verify/* (load on Enable, score on Cross Verify, SSE-streamed). /search stays ~255 ms.
        latency_ms = int((time.perf_counter() - t0) * 1000)
        report = _finalize(t, t0, "search")
        return SearchResponse(
            hits=[Citation(**h) for h in hits],
            latency_ms=latency_ms,
            refused=False,
            timings=report if include_timings else None,
        )
    except HTTPException:
        _finalize(t, t0, "search")  # log partial timings even on upstream error
        raise
    finally:
        timing.reset()


@app.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
def chat(req: ChatRequest, include_timings: bool = False) -> ChatResponse:
    t = timing.start()
    t0 = time.perf_counter()
    try:
        try:
            hits = retrieval.search_with_rerank(req.question, fetch_k=max(20, req.k * 4), return_k=req.k)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"retrieval failed: {e}") from e

        if guardrails.should_refuse([h.get("dense_score", h["score"]) for h in hits]):
            report = _finalize(t, t0, "chat")
            return ChatResponse(
                answer=guardrails.REFUSAL_TEXT,
                citations=[],
                latency_ms=int((time.perf_counter() - t0) * 1000),
                refused=True,
                timings=report if include_timings else None,
            )

        try:
            answer, used_hits = generation.answer_with_citations(req.question, hits)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"generation failed: {e}") from e

        refused = guardrails.is_refusal(answer)
        report = _finalize(t, t0, "chat")
        return ChatResponse(
            answer=answer,
            citations=[] if refused else [Citation(**h) for h in used_hits],
            latency_ms=int((time.perf_counter() - t0) * 1000),
            refused=refused,
            timings=report if include_timings else None,
        )
    except HTTPException:
        _finalize(t, t0, "chat")  # log partial timings even on a downstream error
        raise
    finally:
        timing.reset()


# ── Result Cross Verification (opt-in, streaming) ────────────────────────────
# The CE is CPU-bound (~1 s/pair) on the free tier, so it is OFF the default path: loaded on
# Enable, scored on Cross Verify with each result streamed via SSE, halted by client disconnect.

def _sse(obj: dict) -> str:
    """Format one server-sent event."""
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/verify/enable")
async def verify_enable() -> dict:
    """Lazy-load the CE model (the one-time memory + load-time cost). Idempotent."""
    try:
        await run_in_threadpool(ce_verify.load)  # blocking load off the event loop
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"CE load failed: {e}") from e
    return {"loaded": ce_verify.is_loaded()}


@app.post("/verify/disable")
async def verify_disable() -> dict:
    """Unload the CE model and free its memory (Disable / page-close)."""
    ce_verify.unload()
    return {"loaded": ce_verify.is_loaded()}


@app.get("/verify/stream")
async def verify_stream(request: Request, query: str, k: int = 5) -> StreamingResponse:
    """Stream a calibrated label per result (rank order) for the CURRENT query, one at a time.

    Re-runs the same retrieval (so it scores exactly the post-abstention ranked list), CE-scores
    each result, and emits an SSE `result` event as each is computed. Greyed ("Not relevant",
    CE<=0) results trigger backfill to keep ~TARGET_NONGREY non-grey, hard-capped at MAX_SCORE.
    The loop checks `request.is_disconnected()` BETWEEN each result, so closing the EventSource
    (Stop / navigate away) halts the scoring and frees the CPU.
    """
    async def gen():
        if not ce_verify.is_loaded():
            yield _sse({"event": "error", "reason": "not_loaded"})
            return
        # Re-run retrieval to get the ranked, post-abstention candidate pool (up to MAX_SCORE).
        try:
            hits = await run_in_threadpool(
                retrieval.search_with_rerank, query,
                max(20, ce_verify.MAX_SCORE * 2), ce_verify.MAX_SCORE,
            )
        except Exception as e:  # noqa: BLE001
            yield _sse({"event": "error", "reason": f"retrieval_failed: {e}"})
            return
        dense_scores = [float(h.get("dense_score", h.get("score", 0.0))) for h in hits]
        if not hits or guardrails.should_refuse(dense_scores):
            yield _sse({"event": "done", "reason": "refused", "scored": 0})
            return
        if not ce_verify.applies(query, hits):
            yield _sse({"event": "not_applicable", "reason": "lookup_or_high_confidence"})
            yield _sse({"event": "done", "scored": 0})
            return

        nongrey = 0
        scored = 0
        for rank, h in enumerate(hits, start=1):
            if await request.is_disconnected():
                break  # HALT — client closed the stream; stop burning CPU
            if scored >= ce_verify.MAX_SCORE:
                break  # hard cap
            if rank > k and nongrey >= ce_verify.TARGET_NONGREY:
                break  # initial k scored AND backfill target met
            ce = await run_in_threadpool(ce_verify.score, query, h.get("snippet", "") or "")
            scored += 1
            dense = float(h.get("dense_score", h.get("score", 0.0)))
            lab = ce_verify.label_for(dense, ce)
            if lab["color"] != "grey":
                nongrey += 1
            yield _sse({
                "event": "result", "rank": rank, "result_id": h.get("case_link"),
                "label": lab["label"], "color": lab["color"], "treatment": lab["treatment"],
                "ce_score": round(ce, 3), "dense_score": round(dense, 3),
                "case_link": h.get("case_link"), "snippet": h.get("snippet"),
                "page": h.get("page"), "case_disposition": h.get("case_disposition", ""),
                "case_pub_status": h.get("case_pub_status", ""),
            })
        yield _sse({"event": "done", "scored": scored, "nongrey": nongrey})

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
