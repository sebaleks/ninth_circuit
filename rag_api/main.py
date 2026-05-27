"""FastAPI app for the asylum-case RAG system.

Endpoints:
  POST /chat    — question → answer with citations
  POST /search  — query    → top-k similar cases (no LLM)
  GET  /health  — liveness + index stats
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rag_api import generation, guardrails, nvidia_client, retrieval
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
        embed_model=nvidia_client.EMBED_MODEL,
        rerank_model=nvidia_client.RERANK_MODEL,
        gen_model=nvidia_client.GEN_MODEL,
        build_sha=os.environ.get("RENDER_GIT_COMMIT", "dev")[:7],
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()
    try:
        hits = retrieval.search_with_rerank(req.query, fetch_k=max(20, req.k * 2), return_k=req.k)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e
    latency_ms = int((time.perf_counter() - t0) * 1000)

    # Apply the same dense-score refusal threshold as /chat so out-of-corpus
    # queries don't return noise. The dense score is more uniformly
    # calibrated than the rerank sigmoid.
    if guardrails.should_refuse([h.get("dense_score", h["score"]) for h in hits]):
        return SearchResponse(hits=[], latency_ms=latency_ms, refused=True)

    return SearchResponse(
        hits=[Citation(**h) for h in hits],
        latency_ms=latency_ms,
        refused=False,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    try:
        hits = retrieval.search_with_rerank(req.question, fetch_k=max(20, req.k * 4), return_k=req.k)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"retrieval failed: {e}") from e

    if guardrails.should_refuse([h.get("dense_score", h["score"]) for h in hits]):
        return ChatResponse(
            answer=guardrails.REFUSAL_TEXT,
            citations=[],
            latency_ms=int((time.perf_counter() - t0) * 1000),
            refused=True,
        )

    try:
        answer, used_hits = generation.answer_with_citations(req.question, hits)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"generation failed: {e}") from e

    refused = guardrails.is_refusal(answer)
    return ChatResponse(
        answer=answer,
        citations=[] if refused else [Citation(**h) for h in used_hits],
        latency_ms=int((time.perf_counter() - t0) * 1000),
        refused=refused,
    )
