"""FAISS retrieval + NVIDIA rerank pipeline."""

from __future__ import annotations

import re
from pathlib import Path

import faiss  # type: ignore
import pandas as pd

from rag_api import nvidia_client

# Resolve from repo root regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = _REPO_ROOT / "data" / "index.faiss"
META_PATH = _REPO_ROOT / "data" / "metadata.parquet"

# Module-level singletons loaded once at import (FastAPI startup)
INDEX: faiss.Index | None = None
META: pd.DataFrame | None = None

# Keyword-boost weight: pure-dense retrieval can miss obvious literal matches
# (e.g. query "Honduras" surfacing passages that don't contain the word).
# We add this per-query-token-found to the rerank sigmoid score before sorting,
# which gently floors any chunk that contains the user's literal terms.
# Tuning: the rerank sigmoid is typically 0.001-0.99; 0.10 / token is enough
# to lift literal matches above pure-semantic near-misses without overwhelming
# strong semantic signals.
KEYWORD_BOOST = 0.10
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


def _keyword_boost(query_tokens: list[str], snippet: str) -> float:
    if not query_tokens:
        return 0.0
    lower = snippet.lower()
    return KEYWORD_BOOST * sum(1 for t in query_tokens if t in lower)


def load() -> None:
    """Load FAISS index and metadata into module globals. Called at FastAPI startup."""
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


def search_with_rerank(query: str, fetch_k: int = 20, return_k: int = 5) -> list[dict]:
    """Dense retrieve top-`fetch_k`, then rerank, then keyword-boost, then dedupe by case.

    Each returned hit has:
      - `dense_score`: original FAISS cosine (used for refusal threshold)
      - `score`: rerank sigmoid + per-token keyword boost (ordering signal)

    Returns at most `return_k` results, one per unique case_link (the
    highest-scoring chunk per case is kept).
    """
    # Fetch a wider pool so we still have enough unique cases after dedupe
    pool_size = max(fetch_k, return_k * 5)
    hits = search_dense(query, k=pool_size)
    if not hits:
        return []

    # Preserve the dense cosine separately before overwriting with rerank score
    for h in hits:
        h["dense_score"] = h["score"]

    rerank_scores = nvidia_client.rerank(query, [h["snippet"] for h in hits])
    tokens = _query_tokens(query)
    for h, s in zip(hits, rerank_scores):
        h["score"] = float(s) + _keyword_boost(tokens, h["snippet"])

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
