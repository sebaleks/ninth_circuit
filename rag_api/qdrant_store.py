"""QdrantStore: a VectorStore backed by a Qdrant collection.

Implements the same duck-typed VectorStore protocol as FaissStore (see
rag_api/retrieval.py) and returns the identical hit-dict shape, so the rest of
the pipeline (rerank + BM25 hybrid, dedup) is unchanged.

Metadata lives in each point's payload (written at migration time), so no
separate metadata frame is needed at query time. The point id IS the chunk_id
(also mirrored in the payload), preserving the `chunk_id == META row position`
contract that the local BM25 path depends on.

qdrant-client is imported LAZILY (only in from_env) so importing this module —
and therefore the rest of rag_api — does not require qdrant-client unless the
qdrant backend is actually selected. The search() path uses the injected client
object, so unit tests can pass a fake client without the library installed.
"""

from __future__ import annotations

import os

import numpy as np

# Payload keys stored per point — mirror FaissStore.search's returned hit dict.
PAYLOAD_FIELDS = (
    "chunk_id", "case_link", "snippet", "page", "case_pub_status", "case_disposition",
)


class QdrantStore:
    """Dense retrieval over a Qdrant collection (cosine distance)."""

    name = "qdrant"

    def __init__(self, client, collection: str) -> None:
        self._client = client
        self._collection = collection
        info = client.get_collection(collection)
        self._dim = _collection_dim(info)
        self._ntotal = int(getattr(info, "points_count", 0) or 0)

    @classmethod
    def from_env(cls, collection: str) -> "QdrantStore":
        """Build from QDRANT_URL / QDRANT_API_KEY env vars (qdrant-client lazy-imported)."""
        from qdrant_client import QdrantClient  # lazy: only needed for the qdrant backend

        url = os.environ.get("QDRANT_URL")
        if not url:
            raise RuntimeError("QDRANT_URL not set (required for VECTOR_STORE=qdrant)")
        client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
        return cls(client, collection)

    @property
    def ntotal(self) -> int:
        return int(self._ntotal)

    @property
    def dim(self) -> int:
        return int(self._dim)

    def search(self, query_vec: np.ndarray, k: int) -> list[dict]:
        """ANN search, returning hit dicts identical in shape to FaissStore.search.

        `score` is Qdrant's cosine similarity (the collection is created with
        COSINE distance over L2-normalized vectors), matching FaissStore's
        inner-product-on-normalized-vectors cosine.
        """
        vec = query_vec[0] if getattr(query_vec, "ndim", 1) == 2 else query_vec
        vec = np.asarray(vec, dtype=np.float32).tolist()
        resp = self._client.query_points(
            collection_name=self._collection,
            query=vec,
            limit=k,
            with_payload=True,
        )
        hits: list[dict] = []
        for point in resp.points:
            p = point.payload or {}
            hits.append({
                "chunk_id":         int(p["chunk_id"]),
                "case_link":        str(p["case_link"]),
                "snippet":          str(p["snippet"]),
                "page":             int(p["page"]),
                "score":            float(point.score),
                "case_pub_status":  str(p.get("case_pub_status", "")),
                "case_disposition": str(p.get("case_disposition", "")),
            })
        return hits


def _collection_dim(info) -> int:
    """Best-effort read of the collection's vector size from a CollectionInfo.

    Handles the unnamed-vector case (config.params.vectors is VectorParams with
    .size) and the named-vector case (a dict of VectorParams). Returns 0 if the
    shape can't be parsed (kept defensive — verified against the live client when
    the qdrant backend is wired up)."""
    try:
        vectors = info.config.params.vectors
        if hasattr(vectors, "size"):
            return int(vectors.size)
        return int(next(iter(vectors.values())).size)  # named vectors
    except Exception:  # noqa: BLE001 — telemetry only; never break a request
        return 0
