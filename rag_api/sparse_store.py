"""SparseBM25Store: BM25 retrieval over a Qdrant sparse collection (Path B — BM25 off-box).

Mirrors QdrantStore's lazy-client pattern. `topk()` embeds the query with fastembed's
`Qdrant/bm25` (snowball stemming + IDF — re-validated to reproduce/beat the in-memory rank_bm25
on lookups; see CONFIDENCE_MEMORY_FINDINGS.md "STEP 0") and returns BM25's OWN top-k chunks as
(chunk_id, score). The point id IS the chunk_id (== dense collection id == metadata row), so
results fuse with the dense store by chunk_id without any in-process corpus.

Moving BM25 off-box is what lets the box drop the in-memory BM25 index AND the metadata parquet,
freeing the headroom the cross-encoder needs to fit the 512 MB tier.
"""
from __future__ import annotations

import os


class SparseBM25Store:
    name = "qdrant_bm25"

    def __init__(self, client, collection: str, model=None) -> None:
        self._client = client
        self._collection = collection
        self._model = model  # lazy fastembed SparseTextEmbedding

    @classmethod
    def from_env(cls, collection: str) -> "SparseBM25Store":
        """Build from QDRANT_URL / QDRANT_API_KEY (qdrant-client lazy-imported)."""
        from qdrant_client import QdrantClient  # lazy

        url = os.environ.get("QDRANT_URL")
        if not url:
            raise RuntimeError("QDRANT_URL not set (required for BM25_BACKEND=qdrant)")
        client = QdrantClient(
            url=url, api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=int(os.environ.get("QDRANT_TIMEOUT", "120")),
        )
        return cls(client, collection)

    def _embed(self, query: str):
        if self._model is None:
            from fastembed import SparseTextEmbedding  # lazy: only for the qdrant BM25 backend

            self._model = SparseTextEmbedding(
                model_name=os.environ.get("BM25_MODEL", "Qdrant/bm25")
            )
        return next(iter(self._model.query_embed([query])))

    def topk(self, query: str, k: int) -> list[tuple[int, float]]:
        """BM25's own top-k over the sparse collection, as (chunk_id, score), best first."""
        from qdrant_client import models  # lazy

        qv = self._embed(query)
        resp = self._client.query_points(
            collection_name=self._collection,
            query=models.SparseVector(indices=qv.indices.tolist(), values=qv.values.tolist()),
            using="bm25",
            limit=k,
            with_payload=False,
        )
        return [(int(p.id), float(p.score)) for p in resp.points]
