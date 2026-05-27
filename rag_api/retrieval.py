"""FAISS retrieval + NVIDIA rerank pipeline."""

from __future__ import annotations

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
    """Dense retrieve top-`fetch_k`, then rerank to top-`return_k`.

    Each hit dict has BOTH:
      - `dense_score`: original FAISS cosine (used for refusal threshold)
      - `score`: rerank sigmoid (used for ordering + shown in API response)
    """
    hits = search_dense(query, k=fetch_k)
    if not hits:
        return []
    # Preserve the dense cosine separately before overwriting with rerank score
    for h in hits:
        h["dense_score"] = h["score"]
    rerank_scores = nvidia_client.rerank(query, [h["snippet"] for h in hits])
    for h, s in zip(hits, rerank_scores):
        h["score"] = float(s)
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:return_k]
