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


# ── VectorStore abstraction (FaissStore) ───────────────────────────────────

def test_faiss_store_join_and_padding():
    """FaissStore owns the metadata join and the -1 padding filter that used to
    live inline in search_dense — same hit shape, same ordering."""
    from rag_api.retrieval import FaissStore

    fake_meta = pd.DataFrame({
        "chunk_id":         [0, 1, 2],
        "case_link":        ["a.pdf", "b.pdf", "c.pdf"],
        "text":             ["alpha", "beta", "gamma"],
        "page":             [1, 2, 3],
        "case_pub_status":  ["Published", "Unpublished", "Published"],
        "case_disposition": ["Denied", "Remanded", "Granted"],
    })

    class FakeIndex:
        ntotal = 3
        d = 8
        def search(self, q, k):
            # row 2 best; -1 padding in the 4th slot must be skipped
            return np.array([[0.9, 0.5, 0.1, -1.0]]), np.array([[2, 1, 0, -1]])

    store = FaissStore(FakeIndex(), fake_meta)
    assert store.ntotal == 3
    assert store.dim == 8

    hits = store.search(np.zeros((1, 8), dtype=np.float32), k=4)
    assert len(hits) == 3  # padding row dropped
    assert hits[0]["case_link"] == "c.pdf"
    assert hits[0]["snippet"] == "gamma"
    assert hits[0]["page"] == 3
    assert hits[0]["score"] == pytest.approx(0.9)
    assert hits[0]["case_disposition"] == "Granted"
    assert hits[2]["case_link"] == "a.pdf"
    # chunk_id is the metadata row's chunk_id (the BM25-ordering contract)
    assert [h["chunk_id"] for h in hits] == [2, 1, 0]


def test_faiss_store_size_mismatch_raises():
    """The index/metadata size-mismatch check moved from load() into FaissStore."""
    from rag_api.retrieval import FaissStore

    class FakeIndex:
        ntotal = 5
        d = 8
    meta = pd.DataFrame({"chunk_id": [0, 1], "case_link": ["a", "b"], "text": ["x", "y"],
                         "page": [1, 1], "case_pub_status": ["", ""], "case_disposition": ["", ""]})
    with pytest.raises(RuntimeError, match="size mismatch"):
        FaissStore(FakeIndex(), meta)


def test_search_dense_delegates_to_explicit_store(monkeypatch):
    """When retrieval.STORE is set, search_dense routes through it (the stable
    seam), not through INDEX/META directly."""
    from rag_api import retrieval

    sentinel = [{"chunk_id": 7, "case_link": "z.pdf", "snippet": "s", "page": 1,
                 "score": 0.42, "case_pub_status": "", "case_disposition": ""}]

    class FakeStore:
        ntotal = 1
        dim = 8
        def search(self, q, k):
            return sentinel

    monkeypatch.setattr(retrieval, "STORE", FakeStore())
    monkeypatch.setattr(retrieval, "_embed_query", lambda query: np.zeros((1, 8), dtype=np.float32))
    assert retrieval.search_dense("q", k=5) is sentinel
    assert retrieval.n_chunks() == 1  # n_chunks reads STORE.ntotal


# ── VectorStore abstraction (QdrantStore) ──────────────────────────────────

def _fake_qdrant_client(dim=8, points_count=2):
    """Stand-in for qdrant_client.QdrantClient — no library needed (search uses
    the injected client; only from_env imports qdrant_client)."""
    from types import SimpleNamespace
    info = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=dim))),
        points_count=points_count,
    )

    class _Client:
        def get_collection(self, name):
            return info

        def query_points(self, collection_name, query, limit, with_payload):
            pts = [
                SimpleNamespace(score=0.9, payload={
                    "chunk_id": 2, "case_link": "c.pdf", "snippet": "gamma", "page": 3,
                    "case_pub_status": "Published", "case_disposition": "Granted"}),
                SimpleNamespace(score=0.5, payload={
                    "chunk_id": 1, "case_link": "b.pdf", "snippet": "beta", "page": 2,
                    "case_pub_status": "", "case_disposition": ""}),
            ]
            return SimpleNamespace(points=pts[:limit])

    return _Client()


def test_qdrant_store_search_shape_and_mapping():
    """QdrantStore.search maps payload+score into the FaissStore hit-dict shape."""
    from rag_api.qdrant_store import QdrantStore

    store = QdrantStore(_fake_qdrant_client(dim=8, points_count=2), "test")
    assert store.name == "qdrant"
    assert store.dim == 8
    assert store.ntotal == 2

    hits = store.search(np.zeros((1, 8), dtype=np.float32), k=2)
    assert len(hits) == 2
    assert hits[0]["chunk_id"] == 2 and hits[0]["case_link"] == "c.pdf"
    assert hits[0]["snippet"] == "gamma" and hits[0]["page"] == 3
    assert hits[0]["score"] == pytest.approx(0.9)
    assert hits[0]["case_disposition"] == "Granted"
    assert hits[1]["chunk_id"] == 1 and hits[1]["case_pub_status"] == ""


def test_qdrant_and_faiss_hit_keys_identical():
    """The two stores must emit identical hit-dict key sets (drop-in parity)."""
    from rag_api.retrieval import FaissStore
    from rag_api.qdrant_store import QdrantStore

    fake_meta = pd.DataFrame({
        "chunk_id": [0], "case_link": ["a.pdf"], "text": ["alpha"], "page": [1],
        "case_pub_status": [""], "case_disposition": [""],
    })

    class FakeIndex:
        ntotal = 1
        d = 8
        def search(self, q, k):
            return np.array([[0.9]]), np.array([[0]])

    qv = np.zeros((1, 8), dtype=np.float32)
    faiss_keys = set(FaissStore(FakeIndex(), fake_meta).search(qv, k=1)[0])
    qdrant_keys = set(QdrantStore(_fake_qdrant_client(), "test").search(qv, k=1)[0])
    assert faiss_keys == qdrant_keys


# ── Query-token extraction + BM25 hybrid (no mocks) ────────────────────────

def test_query_tokens_strips_stopwords_and_short():
    from rag_api.retrieval import _query_tokens
    assert _query_tokens("What cases from Honduras?") == ["honduras"]
    assert _query_tokens("the and for") == []           # all stopwords
    assert _query_tokens("a") == []                     # too short
    assert _query_tokens("gang persecution") == ["gang", "persecution"]


def test_bm25_scores_normalize_to_unit(monkeypatch):
    """BM25 per-query scores are normalized so max == 1 (or all zeros)."""
    from rag_api import retrieval
    import rag_api.retrieval as r

    fake_meta = pd.DataFrame({
        "chunk_id":         [0, 1, 2],
        "case_link":        ["a.pdf", "b.pdf", "c.pdf"],
        "text":             [
            "natives and citizens of Honduras seek asylum",
            "the court considered the petition",
            "Honduras is the country of origin",
        ],
        "page":             [1, 1, 1],
        "case_pub_status":  [""] * 3,
        "case_disposition": [""] * 3,
    })
    monkeypatch.setattr(retrieval, "META", fake_meta)
    # Force the cached BM25 to rebuild against the fake corpus
    monkeypatch.setattr(retrieval, "_BM25", None)
    monkeypatch.setattr(retrieval, "_BM25_META_ID", None)

    scores = r._bm25_scores_for_query("honduras")
    assert scores.shape == (3,)
    # At least one chunk should hit the keyword; max is normalized to 1.0
    assert scores.max() == pytest.approx(1.0)
    # The "petition" chunk doesn't contain "honduras"
    assert scores[1] == 0.0


def test_bm25_scores_all_zero_for_empty_token_query(monkeypatch):
    from rag_api import retrieval
    import rag_api.retrieval as r

    fake_meta = pd.DataFrame({
        "chunk_id":         [0, 1],
        "case_link":        ["a.pdf", "b.pdf"],
        "text":             ["foo bar", "baz qux"],
        "page":             [1, 1],
        "case_pub_status":  [""] * 2,
        "case_disposition": [""] * 2,
    })
    monkeypatch.setattr(retrieval, "META", fake_meta)
    monkeypatch.setattr(retrieval, "_BM25", None)
    monkeypatch.setattr(retrieval, "_BM25_META_ID", None)

    # All tokens stripped as stopwords → no signal
    scores = r._bm25_scores_for_query("the and for")
    assert (scores == 0.0).all()


# ── search_with_rerank: dedup + hybrid (rerank + BM25) integration ──────────

def test_search_with_rerank_dedupes_by_case_link(monkeypatch):
    """Multiple chunks from the same case collapse to one (highest-scoring) hit."""
    from rag_api import retrieval, nvidia_client

    # 4 chunks from 2 cases (case A has 3 chunks, case B has 1)
    fake_meta = pd.DataFrame({
        "chunk_id":         [0, 1, 2, 3],
        "case_link":        ["A.pdf", "A.pdf", "A.pdf", "B.pdf"],
        "text":             ["a1", "a2", "a3", "b1"],
        "page":             [1, 2, 3, 1],
        "case_pub_status":  [""] * 4,
        "case_disposition": [""] * 4,
    })

    class FakeIndex:
        ntotal = 4
        def search(self, q, k):
            # Return all 4 in order by chunk_id
            return np.array([[0.5, 0.45, 0.4, 0.35]]), np.array([[0, 1, 2, 3]])

    monkeypatch.setattr(retrieval, "INDEX", FakeIndex())
    monkeypatch.setattr(retrieval, "META", fake_meta)
    monkeypatch.setattr(retrieval, "_BM25", None)         # force rebuild on this corpus
    monkeypatch.setattr(retrieval, "_BM25_META_ID", None)
    monkeypatch.setattr(
        nvidia_client, "embed_query",
        lambda text: np.zeros((1, 2048), dtype=np.float32),
    )
    # Rerank returns scores in same order: A's a1 best, then a2, a3, b1
    monkeypatch.setattr(
        nvidia_client, "rerank",
        lambda q, passages: [0.8, 0.7, 0.6, 0.5],
    )

    # Query has no overlap with corpus → BM25 contributes 0, rerank fully decides
    hits = retrieval.search_with_rerank("xyzzy", fetch_k=4, return_k=5)
    # Should return only 2 hits (one per case), not 4
    assert len(hits) == 2
    case_links = [h["case_link"] for h in hits]
    assert case_links == ["A.pdf", "B.pdf"]
    # The kept chunk from case A is the highest-scoring one (a1)
    assert hits[0]["snippet"] == "a1"


def test_search_with_rerank_bm25_lifts_literal_matches(monkeypatch):
    """BM25 hybrid scoring surfaces passages that literally contain the query word
    above semantic near-misses, fixing the 'Honduras → top-hit-has-no-Honduras' bug.

    Note: BM25 IDF needs a corpus of meaningful size — with only 2 docs and the
    query term in 1, IDF collapses to ~0. So we seed a small corpus with several
    decoy chunks that don't contain the keyword.
    """
    from rag_api import retrieval, nvidia_client

    # 6 chunks total: only B.pdf's chunk (#5) contains "honduras". Decoy chunks
    # give BM25 enough corpus stats to make "honduras" a high-IDF term.
    fake_meta = pd.DataFrame({
        "chunk_id":         list(range(6)),
        "case_link":        ["A.pdf"] + ["decoy.pdf"] * 4 + ["B.pdf"],
        "text":             [
            "the court denied the petition for review",            # A.pdf — top dense hit, no "Honduras"
            "credibility findings against the petitioner",
            "withholding of removal granted in part",
            "internal relocation reasonable in this case",
            "the government appealed the immigration ruling",
            "natives and citizens of Honduras seek asylum",        # B.pdf — mentions Honduras
        ],
        "page":             [1] * 6,
        "case_pub_status":  [""] * 6,
        "case_disposition": [""] * 6,
    })

    class FakeIndex:
        ntotal = 6
        def search(self, q, k):
            # Only A.pdf and B.pdf survive FAISS top-2; the decoys score below
            return np.array([[0.5, 0.45]]), np.array([[0, 5]])

    monkeypatch.setattr(retrieval, "INDEX", FakeIndex())
    monkeypatch.setattr(retrieval, "META", fake_meta)
    monkeypatch.setattr(retrieval, "_BM25", None)
    monkeypatch.setattr(retrieval, "_BM25_META_ID", None)
    monkeypatch.setattr(
        nvidia_client, "embed_query",
        lambda text: np.zeros((1, 2048), dtype=np.float32),
    )
    # Rerank prefers A.pdf (semantic match without the keyword). BM25 should
    # overpower this because "honduras" is in B.pdf but not A.pdf, and BM25
    # gives "honduras" a high IDF (rare in the corpus).
    monkeypatch.setattr(
        nvidia_client, "rerank",
        lambda q, passages: [0.05, 0.02],
    )

    hits = retrieval.search_with_rerank("Honduras", fetch_k=2, return_k=2)
    # BM25 contribution flips the order: B.pdf above A.pdf
    assert hits[0]["case_link"] == "B.pdf"
    assert hits[1]["case_link"] == "A.pdf"
    # BM25 + rerank are both present on the hit dict
    assert "rerank_score" in hits[0]
    assert "bm25_score" in hits[0]
    assert hits[0]["bm25_score"] == pytest.approx(1.0)  # normalized max
    assert hits[1]["bm25_score"] == pytest.approx(0.0)  # no "honduras"


# ── Latency instrumentation ─────────────────────────────────────────────────

def _wire_fake_pipeline(monkeypatch):
    """Configure retrieval with in-memory fakes so the full search_with_rerank
    pipeline runs through every instrumented stage (embed → dense → rerank →
    bm25 → fusion) without network or the data/ artifacts. Returns nothing; the
    caller drives it via TestClient."""
    from rag_api import retrieval, nvidia_client

    fake_meta = pd.DataFrame({
        "chunk_id":         [0, 1],
        "case_link":        ["a.pdf", "b.pdf"],
        "text":             ["asylum persecution honduras", "withholding removal credibility"],
        "page":             [1, 2],
        "case_pub_status":  ["Published", "Unpublished"],
        "case_disposition": ["Denied", "Granted"],
    })

    class FakeStore:
        name = "faiss"
        ntotal = 2
        dim = 8
        def search(self, q, k):
            # dense scores comfortably above the refusal threshold
            return [
                {"chunk_id": 0, "case_link": "a.pdf", "snippet": "asylum persecution honduras",
                 "page": 1, "score": 0.8, "case_pub_status": "Published", "case_disposition": "Denied"},
                {"chunk_id": 1, "case_link": "b.pdf", "snippet": "withholding removal credibility",
                 "page": 2, "score": 0.7, "case_pub_status": "Unpublished", "case_disposition": "Granted"},
            ]

    class FakeEmbedder:
        model_name = "fake/local-embedder"
        def embed_query(self, text):
            return np.zeros((1, 8), dtype=np.float32)

    monkeypatch.setattr(retrieval, "load", lambda: None)
    monkeypatch.setattr(retrieval, "STORE", FakeStore())
    monkeypatch.setattr(retrieval, "META", fake_meta)
    monkeypatch.setattr(retrieval, "EMBEDDER", FakeEmbedder())
    monkeypatch.setattr(retrieval, "_BM25", None)
    monkeypatch.setattr(retrieval, "_BM25_META_ID", None)
    monkeypatch.setattr(nvidia_client, "rerank", lambda q, passages: [0.6, 0.5])


def _post(path, monkeypatch, **json_body):
    from fastapi.testclient import TestClient
    from rag_api.main import app
    with TestClient(app) as client:
        return client.post(path, json=json_body)


def test_search_omits_timings_by_default(monkeypatch):
    """Idempotency: without opt-in the response shape is unchanged — no timings key."""
    _wire_fake_pipeline(monkeypatch)
    resp = _post("/search", monkeypatch, query="honduras asylum", k=5)
    assert resp.status_code == 200
    body = resp.json()
    assert "timings" not in body  # omitted entirely, not null
    for key in ("hits", "latency_ms", "refused"):
        assert key in body
    assert len(body["hits"]) == 2


def test_search_timings_present_and_context_populated(monkeypatch):
    """With include_timings=true every expected stage + context key is present."""
    _wire_fake_pipeline(monkeypatch)
    resp = _post("/search?include_timings=true", monkeypatch, query="honduras asylum", k=5)
    assert resp.status_code == 200
    t = resp.json()["timings"]

    for key in ("embed_ms", "dense_search_ms", "rerank_ms", "bm25_ms",
                "fusion_dedup_ms", "server_total_ms",
                "embed_retry_sleep_ms", "embed_attempts",
                "rerank_retry_sleep_ms", "rerank_attempts",
                "embedder_name", "rerank_pool_size", "vector_store", "embedding_dim"):
        assert key in t, f"missing timing key: {key}"

    assert "generate_ms" not in t  # /search never generates
    # Context fields populate correctly
    assert t["embedder_name"] == "fake/local-embedder"
    assert t["vector_store"] == "faiss"
    assert t["embedding_dim"] == 8
    assert t["rerank_pool_size"] == 2
    # No retries occurred in the fakes → attempts == 1, sleep == 0
    assert t["embed_attempts"] == 1
    assert t["rerank_retry_sleep_ms"] == 0.0


def test_search_timings_sum_to_server_total(monkeypatch):
    """Reported components sum to ~server_total_ms (fusion_dedup absorbs the rest)."""
    _wire_fake_pipeline(monkeypatch)
    resp = _post("/search?include_timings=true", monkeypatch, query="honduras asylum", k=5)
    t = resp.json()["timings"]
    components = (t["embed_ms"] + t["embed_retry_sleep_ms"]
                 + t["dense_search_ms"]
                 + t["rerank_ms"] + t["rerank_retry_sleep_ms"]
                 + t["bm25_ms"]
                 + t["fusion_dedup_ms"])
    assert abs(components - t["server_total_ms"]) <= 5.0


def test_chat_timings_include_generate(monkeypatch):
    """/chat adds the generate stage with its own attempts/sleep keys."""
    _wire_fake_pipeline(monkeypatch)
    from rag_api import nvidia_client
    monkeypatch.setattr(nvidia_client, "generate",
                        lambda q, passages, max_tokens=500: "The court denied the petition [1].")
    resp = _post("/chat?include_timings=true", monkeypatch, question="why was asylum denied?", k=3)
    assert resp.status_code == 200
    body = resp.json()
    t = body["timings"]
    for key in ("generate_ms", "generate_retry_sleep_ms", "generate_attempts"):
        assert key in t, f"missing timing key: {key}"
    assert t["generate_attempts"] == 1
    # full-pipeline sum still closes within tolerance
    components = (t["embed_ms"] + t["embed_retry_sleep_ms"] + t["dense_search_ms"]
                 + t["rerank_ms"] + t["rerank_retry_sleep_ms"] + t["bm25_ms"]
                 + t["generate_ms"] + t["generate_retry_sleep_ms"] + t["fusion_dedup_ms"])
    assert abs(components - t["server_total_ms"]) <= 5.0


def test_chat_default_omits_timings(monkeypatch):
    _wire_fake_pipeline(monkeypatch)
    from rag_api import nvidia_client
    monkeypatch.setattr(nvidia_client, "generate",
                        lambda q, passages, max_tokens=500: "Denied [1].")
    resp = _post("/chat", monkeypatch, question="why was asylum denied?", k=3)
    assert resp.status_code == 200
    assert "timings" not in resp.json()


def test_retry_sleep_excluded_from_network_stage(monkeypatch):
    """A 429 → retry → success must NOT inflate embed_ms; the backoff lands in
    embed_retry_sleep_ms instead, and embed_attempts counts the retry."""
    from rag_api import nvidia_client, timing

    # Don't actually wait out the backoff; record_sleep still logs the nominal 2s.
    monkeypatch.setattr(nvidia_client.time, "sleep", lambda s: None)

    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise nvidia_client._RetryableNimError(429, RuntimeError("rate limited"))
        return "ok"

    timing.start()
    try:
        with timing.timer("embed"):  # the gross dispatch-point timer
            result = nvidia_client._with_retry("embed", flaky)
        report = timing.build_report(timing.current(), server_total_ms=1000.0)
    finally:
        timing.reset()

    assert result == "ok"
    assert calls["n"] == 2
    assert report["embed_attempts"] == 2
    assert report["embed_retry_sleep_ms"] == 2000.0     # first backoff = 2s
    # Sleep excluded: gross (tiny, sleep mocked) minus 2000ms clamps to 0.
    assert report["embed_ms"] == 0.0


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
