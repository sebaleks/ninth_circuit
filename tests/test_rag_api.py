"""Unit tests for the RAG API.

These run on every PR via .github/workflows/rag-api-test.yml. They mock out the
NVIDIA NIM calls and the FAISS index so they pass without network access or
needing the data/index.faiss artifact materialized — which keeps the CI run
fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Citation parsing — pure function, no mocks needed ────────────────────────

def test_parse_citations_basic():
    from rag_api.generation import parse_citations
    answer = "The court denied the petition [1] and granted in part [3]."
    assert parse_citations(answer, n_passages=5) == [0, 2]


def test_parse_citations_dedup_and_filter():
    from rag_api.generation import parse_citations
    answer = "Cases [1][1][7] cite this. Also [2]."
    # [1] appears twice (kept once, in first-appearance order); [7] is out of range
    assert parse_citations(answer, n_passages=3) == [0, 1]


def test_parse_citations_none():
    from rag_api.generation import parse_citations
    assert parse_citations("No citations here.", n_passages=5) == []


# ── Guardrails ────────────────────────────────────────────────────────────────

def test_should_refuse_empty():
    from rag_api.guardrails import should_refuse
    assert should_refuse([]) is True


def test_should_refuse_below_threshold():
    from rag_api.guardrails import should_refuse, MIN_DENSE_SCORE
    assert should_refuse([MIN_DENSE_SCORE - 0.01, 0.0, 0.05]) is True


def test_should_refuse_above_threshold():
    from rag_api.guardrails import should_refuse, MIN_DENSE_SCORE
    assert should_refuse([MIN_DENSE_SCORE + 0.01, 0.0]) is False


def test_is_refusal_match():
    from rag_api.guardrails import is_refusal, REFUSAL_TEXT
    assert is_refusal(REFUSAL_TEXT) is True
    assert is_refusal(REFUSAL_TEXT.upper()) is True
    assert is_refusal("The court found ...") is False


# ── Pydantic models ─────────────────────────────────────────────────────────

def test_chat_request_validation():
    from rag_api.models import ChatRequest
    r = ChatRequest(question="hi", k=3)
    assert r.k == 3
    with pytest.raises(Exception):
        ChatRequest(question="", k=3)  # empty question
    with pytest.raises(Exception):
        ChatRequest(question="hi", k=100)  # k too big


def test_citation_accepts_extra_fields():
    """Hits dicts include `dense_score` which Citation should silently ignore."""
    from rag_api.models import Citation
    hit = {
        "chunk_id": 1, "case_link": "https://x", "snippet": "...", "page": 1,
        "score": 0.5, "dense_score": 0.3,
    }
    c = Citation(**hit)
    assert c.score == 0.5
    assert not hasattr(c, "dense_score")  # extra silently dropped


# ── Retrieval with mocked FAISS + NVIDIA ────────────────────────────────────

def test_search_dense_with_mocked_index(monkeypatch):
    """Verify retrieval.search_dense maps FAISS hits → metadata rows correctly."""
    from rag_api import retrieval, nvidia_client

    fake_meta = pd.DataFrame({
        "chunk_id":         [0, 1, 2],
        "case_link":        ["a.pdf", "b.pdf", "c.pdf"],
        "text":             ["alpha snippet", "beta snippet", "gamma snippet"],
        "page":             [1, 2, 3],
        "case_pub_status":  ["Published", "Unpublished", "Published"],
        "case_disposition": ["Denied", "Remanded", "Granted"],
    })

    class FakeIndex:
        ntotal = 3
        def search(self, q, k):
            # Return scores + indices ordered: row 2 best, row 0 worst
            return np.array([[0.9, 0.5, 0.1]]), np.array([[2, 1, 0]])

    monkeypatch.setattr(retrieval, "INDEX", FakeIndex())
    monkeypatch.setattr(retrieval, "META", fake_meta)
    monkeypatch.setattr(
        nvidia_client, "embed_query",
        lambda text: np.zeros((1, 2048), dtype=np.float32),
    )

    hits = retrieval.search_dense("test query", k=3)
    assert len(hits) == 3
    assert hits[0]["case_link"] == "c.pdf"
    assert hits[0]["page"] == 3
    assert hits[0]["score"] == pytest.approx(0.9)
    assert hits[2]["case_link"] == "a.pdf"


def test_search_dense_skips_negative_indices(monkeypatch):
    """FAISS pads with -1 when fewer than k results are available."""
    from rag_api import retrieval, nvidia_client

    fake_meta = pd.DataFrame({
        "chunk_id":         [0],
        "case_link":        ["only.pdf"],
        "text":             ["only snippet"],
        "page":             [1],
        "case_pub_status":  [""],
        "case_disposition": [""],
    })

    class FakeIndex:
        ntotal = 1
        def search(self, q, k):
            return np.array([[0.7, -1.0]]), np.array([[0, -1]])

    monkeypatch.setattr(retrieval, "INDEX", FakeIndex())
    monkeypatch.setattr(retrieval, "META", fake_meta)
    monkeypatch.setattr(
        nvidia_client, "embed_query",
        lambda text: np.zeros((1, 2048), dtype=np.float32),
    )

    hits = retrieval.search_dense("q", k=5)
    assert len(hits) == 1
    assert hits[0]["case_link"] == "only.pdf"


# ── Health endpoint — uses FastAPI TestClient + lifespan ────────────────────

def test_health_endpoint_shape(monkeypatch):
    """/health returns the documented shape even if data/ isn't materialized."""
    from rag_api import retrieval

    # Skip the lifespan FAISS load — we only need to verify the response schema
    monkeypatch.setattr(retrieval, "load", lambda: None)
    monkeypatch.setattr(retrieval, "n_chunks", lambda: 42)

    from fastapi.testclient import TestClient
    from rag_api.main import app

    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("status", "n_chunks", "embed_model", "rerank_model", "gen_model", "build_sha"):
        assert key in body
    assert body["status"] == "ok"
    assert body["n_chunks"] == 42
    assert body["embed_model"].startswith("nvidia/")
