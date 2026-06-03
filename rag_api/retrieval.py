"""FAISS retrieval + NVIDIA rerank + BM25 hybrid pipeline."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import faiss  # type: ignore
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from rag_api import nvidia_client

# Resolve from repo root regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = _REPO_ROOT / "data"
INDEX_PATH = DEFAULT_INDEX_DIR / "index.faiss"
META_PATH = DEFAULT_INDEX_DIR / "metadata.parquet"

# Module-level singletons loaded once at import (FastAPI startup)
INDEX: faiss.Index | None = None
META: pd.DataFrame | None = None
# Query-time embedder. None means "use the NIM path" (nvidia_client.embed_query),
# which is also the default before load() runs — so tests that monkeypatch
# nvidia_client.embed_query keep working without any setup. load() may replace
# this with a LocalEmbedder based on the index's config.json.
EMBEDDER = None
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


def _index_dir() -> Path:
    """Directory to load the index/metadata/config from (env INDEX_DIR, default data/)."""
    return Path(os.environ.get("INDEX_DIR", str(DEFAULT_INDEX_DIR)))


def _resolve_embedder(config_path: Path, index_dim: int):
    """Pick the query-time embedder matching the index, failing loudly on dim mismatch.

    Returns the embedder object (LocalEmbedder), or None to mean the NIM path.
    Backward-compatible: a missing config.json (e.g. the legacy data/ index)
    defaults to NIM.
    """
    if not config_path.exists():
        embedder_name, embedder, expected_dim = "nim", None, nvidia_client.EMBED_DIM
    else:
        cfg = json.loads(config_path.read_text())
        embedder_name = cfg.get("embedder", "nim")
        if embedder_name == "nim":
            embedder, expected_dim = None, nvidia_client.EMBED_DIM
        else:
            from rag_api.local_embedder import LocalEmbedder
            embedder = LocalEmbedder(cfg["model_id"])
            expected_dim = embedder.dim

    if index_dim != expected_dim:
        raise RuntimeError(
            f"Embedder/index dimension mismatch: FAISS index is {index_dim}-dim but "
            f"embedder '{embedder_name}' produces {expected_dim}-dim query vectors. "
            f"Set INDEX_DIR to a matching index, or re-ingest with this embedder."
        )
    return embedder


def load() -> None:
    """Load FAISS index, metadata, embedder, and BM25 corpus. Called at FastAPI startup.

    Reads from INDEX_DIR (env, default data/): index.faiss, metadata.parquet, and
    config.json. The embedder is chosen from config.json so query-time embedding
    matches how the index was built; reranker and generation stay on NIM.
    """
    global INDEX, META, EMBEDDER
    index_dir = _index_dir()
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "metadata.parquet"
    config_path = index_dir / "config.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path} — run `python pipeline/rag_ingest.py` first")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path} — run `python pipeline/rag_ingest.py` first")
    INDEX = faiss.read_index(str(index_path))
    META = pd.read_parquet(meta_path)
    if len(META) != INDEX.ntotal:
        raise RuntimeError(
            f"Index/metadata size mismatch: index={INDEX.ntotal} meta={len(META)}"
        )
    EMBEDDER = _resolve_embedder(config_path, INDEX.d)
    # Pre-warm BM25 so the first query doesn't pay the build cost
    _ensure_bm25()


def _embed_query(query: str) -> np.ndarray:
    """Embed a query with the configured embedder, falling back to the NIM path."""
    if EMBEDDER is None:
        return nvidia_client.embed_query(query)
    return EMBEDDER.embed_query(query)


def n_chunks() -> int:
    if INDEX is None:
        return 0
    return int(INDEX.ntotal)


def search_dense(query: str, k: int = 20) -> list[dict]:
    """Embed query, FAISS top-k, return list of hit dicts with raw cosine score."""
    if INDEX is None or META is None:
        raise RuntimeError("retrieval.load() must be called before search_dense()")

    q_vec = _embed_query(query)  # (1, dim), already L2-normalized
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
