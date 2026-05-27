"""Pydantic request/response schemas for the RAG API."""

from __future__ import annotations

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


class SearchResponse(BaseModel):
    hits: list[Citation]
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    n_chunks: int
    embed_model: str
    rerank_model: str
    gen_model: str
    build_sha: str
