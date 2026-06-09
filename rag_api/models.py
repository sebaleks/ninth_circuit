"""Pydantic request/response schemas for the RAG API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=20, description="number of citations to return")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(10, ge=1, le=50)


class Citation(BaseModel):
    chunk_id: int
    case_link: str
    snippet: str
    page: int
    score: float
    case_pub_status: str = ""
    case_disposition: str = ""


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: int
    refused: bool = False
    # Per-stage latency breakdown; only populated when the request opts in via
    # ?include_timings=true, and omitted from the JSON otherwise (routes set
    # response_model_exclude_none=True). Additive — does not affect latency_ms.
    timings: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    hits: list[Citation]
    latency_ms: int
    refused: bool = False
    timings: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    n_chunks: int
    embedding_dim: int | None = None  # dim of the loaded index (e.g. 2048 / 512 / 384)
    # Actual runtime configuration (reflects env-selected backend + knobs).
    vector_store: str | None = None   # "faiss" | "qdrant"
    embedder: str | None = None       # actual loaded embedder (local model id or NIM model)
    use_reranker: bool | None = None
    fusion_method: str | None = None  # "blend" | "rrf"
    embed_model: str                  # NIM embed model constant (rerank/gen stay on NIM)
    rerank_model: str
    gen_model: str
    build_sha: str
