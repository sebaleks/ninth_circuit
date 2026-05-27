"""FAISS retrieval + NVIDIA rerank + BM25 hybrid pipeline."""

from __future__ import annotations

import re
from pathlib import Path

import faiss  # type: ignore
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from rag_api import nvidia_client

# Resolve from repo root regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = _REPO_ROOT / "data" / "index.faiss"
META_PATH = _REPO_ROOT / "data" / "metadata.parquet"

# Module-level singletons loaded once at import (FastAPI startup)
INDEX: faiss.Index | None = None
META: pd.DataFrame | None = None
_BM25: BM25Okapi | None = None
_BM25_META_ID: int | None = None  # id(META) at the time BM25 was built; for invalidation

# Hybrid weight: final = ALPHA * rerank_sigmoid + (1 - ALPHA) * bm25_normalized
# 0.6 (60% rerank / 40% BM25) is the legal-IR community default for hybrid
# dense+sparse. Tuneable via evaluation — exposed as a constant for ablations.
HYBRID_ALPHA = 0.6

# Tokenizer shared by query and corpus so BM25 sees consistent terms.
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what",
    "where", "when", "which", "who", "whom", "does", "did", "was",
    "were", "are", "have", "has", "had", "but", "not", "all",
    "any", "some", "into", "about", "case", "cases",
}


def _query_tokens(query: str) -> list[str]:
    """Lowercase alphanumeric tokens of length >= 3, excluding common stopwords."""
    return [
        t for t in re.findall(r"[a-z0-9]+", query.lower())
        if len(t) >= 3 and t not in _STOPWORDS
    ]


def _ensure_bm25() -> BM25Okapi | None:
    """Build the BM25 index on the current META if not already cached.

    Returns None if the corpus is degenerate (every doc tokenizes to []),
    which would otherwise crash BM25Okapi with a div-by-zero.

    Caches by id(META) so tests that monkeypatch META get a fresh BM25 for
    free. In production, META is loaded once at startup and never reassigned,
    so the build happens exactly once (cost: <1s for ~700 chunks, ~3s for 36k).
    """
    global _BM25, _BM25_META_ID
    if META is None:
        raise RuntimeError("retrieval.load() must be called before BM25 access")
    if _BM25_META_ID == id(META):
        return _BM25
    corpus_tokens = [_query_tokens(t) for t in META["text"].tolist()]
    if not any(corpus_tokens):
        _BM25 = None  # degenerate; scoring will return zeros
    else:
        _BM25 = BM25Okapi(corpus_tokens)
    _BM25_META_ID = id(META)
    return _BM25


def load() -> None:
    """Load FAISS index, metadata, and BM25 corpus. Called at FastAPI startup."""
    global INDEX, META
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing {INDEX_PATH} — run `python pipeline/rag_ingest.py` first")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing {META_PATH} — run `python pipeline/rag_ingest.py` first")
    INDEX = faiss.read_index(str(INDEX_PATH))
    META = pd.read_parquet(META_PATH)
    if len(META) != INDEX.ntotal:
        raise RuntimeError(
            f"Index/metadata size mismatch: index={INDEX.ntotal} meta={len(META)}"
        )
    # Pre-warm BM25 so the first query doesn't pay the build cost
    _ensure_bm25()


def n_chunks() -> int:
    if INDEX is None:
        return 0
    return int(INDEX.ntotal)


def search_dense(query: str, k: int = 20) -> list[dict]:
    """Embed query, FAISS top-k, return list of hit dicts with raw cosine score."""
    if INDEX is None or META is None:
        raise RuntimeError("retrieval.load() must be called before search_dense()")

    q_vec = nvidia_client.embed_query(query)  # (1, 2048), already L2-normalized
    scores, indices = INDEX.search(q_vec, k)

    hits: list[dict] = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0:  # FAISS pads with -1 when fewer than k results
            continue
        row = META.iloc[idx]
        hits.append({
            "chunk_id":         int(row["chunk_id"]),
            "case_link":        str(row["case_link"]),
            "snippet":          str(row["text"]),
            "page":             int(row["page"]),
            "score":            float(score),
            "case_pub_status":  str(row.get("case_pub_status", "")),
            "case_disposition": str(row.get("case_disposition", "")),
        })
    return hits


def _bm25_scores_for_query(query: str) -> np.ndarray:
    """Per-document BM25 scores for the whole corpus, normalized to [0, 1].

    Normalization is per-query (divide by the max raw score in this query's
    result, if any). This keeps the hybrid combination meaningful regardless
    of how the BM25 raw scale shifts between queries.
    """
    bm25 = _ensure_bm25()
    tokens = _query_tokens(query)
    n = len(META) if META is not None else 0
    if bm25 is None or not tokens:
        return np.zeros(n, dtype=np.float32)
    raw = bm25.get_scores(tokens)
    top = raw.max()
    if top <= 0:
        return np.zeros_like(raw, dtype=np.float32)
    return (raw / top).astype(np.float32)


def search_with_rerank(query: str, fetch_k: int = 20, return_k: int = 5) -> list[dict]:
    """Dense retrieve top-`fetch_k`, then rerank + BM25 hybrid, then dedupe by case.

    Each returned hit has:
      - `dense_score`: original FAISS cosine (used for refusal threshold)
      - `bm25_score`: BM25 score for this chunk, normalized to [0, 1]
      - `rerank_score`: NVIDIA rerank sigmoid
      - `score`: HYBRID_ALPHA * rerank + (1 - HYBRID_ALPHA) * bm25 (ordering signal)

    Returns at most `return_k` results, one per unique case_link (the
    highest-scoring chunk per case is kept).
    """
    # Fetch a wider pool so we still have enough unique cases after dedupe
    pool_size = max(fetch_k, return_k * 5)
    hits = search_dense(query, k=pool_size)
    if not hits:
        return []

    # Preserve the dense cosine separately before overwriting with hybrid score
    for h in hits:
        h["dense_score"] = h["score"]

    # NVIDIA rerank — semantic similarity to the query, returned as sigmoid in [0, 1]
    rerank_scores = nvidia_client.rerank(query, [h["snippet"] for h in hits])

    # BM25 — keyword/IDF scoring, normalized per-query to [0, 1]
    bm25_all = _bm25_scores_for_query(query)
    for h, r_score in zip(hits, rerank_scores):
        b_score = float(bm25_all[h["chunk_id"]])
        h["rerank_score"] = float(r_score)
        h["bm25_score"] = b_score
        h["score"] = HYBRID_ALPHA * float(r_score) + (1.0 - HYBRID_ALPHA) * b_score

    hits.sort(key=lambda h: h["score"], reverse=True)

    # Dedupe: one chunk per case, keep the highest-scoring (first after sort)
    seen: set[str] = set()
    deduped: list[dict] = []
    for h in hits:
        if h["case_link"] in seen:
            continue
        seen.add(h["case_link"])
        deduped.append(h)
        if len(deduped) >= return_k:
            break
    return deduped
