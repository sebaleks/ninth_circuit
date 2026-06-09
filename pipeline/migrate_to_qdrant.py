#!/usr/bin/env python3
"""Migrate a FAISS-backed index into a Qdrant collection.

Reads metadata.parquet (+ config.json) from --index-dir and upserts vectors into
the named Qdrant collection, with the chunk_id as the point id and the retrieval
metadata as payload. The collection is created with COSINE distance (vectors are
L2-normalized), matching the inner-product-on-normalized-vectors cosine used by
FaissStore. chunk_id ordering is preserved (point id == chunk_id == metadata row
position), so the local BM25 path keeps working under the qdrant backend.

Vector source (--source):
  reembed  (DEFAULT) — re-embed the `text` column with the SAME embedder/dim from
                       config.json, producing the EXACT NIM (or local) vectors.
                       This is the faithful path for a genuine FAISS-vs-Qdrant
                       comparison. Requires the embedder (NIM healthy enough to
                       embed ~all chunks at the configured dim).
  reconstruct        — reconstruct vectors from the FAISS index. ⚠️ IVFPQ is LOSSY
                       (stores PQ codes, not originals), so these are APPROXIMATE.
                       Fallback only — e.g. if NIM is unavailable.

Invoked MANUALLY for T3 ingestion — never run automatically.

Usage:
    python pipeline/migrate_to_qdrant.py --index-dir data/experiments/T2 \
        --collection asylum_cases [--source reembed|reconstruct] [--recreate]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import faiss  # type: ignore
import numpy as np
import pandas as pd

# Make the repo root importable so `from rag_api ...` works when this is run as a
# script (python pipeline/migrate_to_qdrant.py), mirroring pipeline/rag_ingest.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

# Payload keys stored per point (mirror FaissStore.search's hit dict). `snippet`
# comes from the parquet `text` column and is handled explicitly.
_PAYLOAD_FIELDS = ("chunk_id", "case_link", "page", "case_pub_status", "case_disposition")
_EMBED_BATCH = 16  # NIM rate-limit friendly, matches rag_ingest


def reembed_vectors(index_dir: Path, meta: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Re-embed meta['text'] (in chunk_id/row order) with the config.json embedder.

    Returns (vectors, dim). Vectors are L2-normalized (both embedder paths return
    normalized output). This reproduces the EXACT vectors the index was built from.
    """
    cfg = json.loads((index_dir / "config.json").read_text())
    embedder_name = cfg.get("embedder", "nim")
    dim = int(cfg.get("dim", 2048))
    texts = meta["text"].astype(str).tolist()  # row order == chunk_id order

    if embedder_name == "nim":
        from rag_api import nvidia_client
        print(f"Re-embedding {len(texts)} chunks via NIM at dim={dim} (batch={_EMBED_BATCH})…")
        parts: list[np.ndarray] = []
        for i in range(0, len(texts), _EMBED_BATCH):
            parts.append(nvidia_client.embed_passages(texts[i : i + _EMBED_BATCH], dim=dim))
            if (i // _EMBED_BATCH) % 5 == 0:
                print(f"  embedded {min(i + _EMBED_BATCH, len(texts))}/{len(texts)}")
        return np.vstack(parts).astype(np.float32), dim

    from rag_api.local_embedder import LocalEmbedder
    emb = LocalEmbedder(cfg["model_id"])
    print(f"Re-embedding {len(texts)} chunks via local {cfg['model_id']} (dim={emb.dim})…")
    return emb.embed_passages(texts).astype(np.float32), emb.dim


def reconstruct_vectors(index: faiss.Index) -> tuple[np.ndarray, int]:
    """Reconstruct all vectors from the (IVFPQ) index, L2-normalized.

    ⚠️ APPROXIMATE for IVFPQ — fallback only (see module docstring)."""
    print(f"Reconstructing {index.ntotal} vectors (dim={index.d}) from FAISS "
          f"— ⚠️ APPROXIMATE (IVFPQ is lossy; prefer --source reembed)")
    index.make_direct_map()
    vecs = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32), index.d


def build_points(vectors: np.ndarray, meta: pd.DataFrame):
    """Yield (id, vector, payload) with id == chunk_id (ordering preserved)."""
    for i, row in enumerate(meta.itertuples(index=False)):
        rec = row._asdict()
        payload = {f: rec[f] for f in _PAYLOAD_FIELDS if f in rec}
        payload["chunk_id"] = int(rec["chunk_id"])
        payload["page"] = int(rec["page"])
        payload["snippet"] = str(rec["text"])
        yield int(rec["chunk_id"]), vectors[i].tolist(), payload


def migrate(index_dir: Path, collection: str, source: str = "reembed",
            batch: int = 256, recreate: bool = False) -> dict:
    from qdrant_client import QdrantClient  # lazy
    from qdrant_client.models import Distance, PointStruct, VectorParams  # lazy

    meta = pd.read_parquet(index_dir / "metadata.parquet")
    index = faiss.read_index(str(index_dir / "index.faiss"))
    if len(meta) != index.ntotal:
        raise RuntimeError(f"index/metadata mismatch: index={index.ntotal} meta={len(meta)}")

    if source == "reembed":
        vectors, dim = reembed_vectors(index_dir, meta)
    elif source == "reconstruct":
        vectors, dim = reconstruct_vectors(index)
    else:
        raise ValueError(f"--source must be 'reembed' or 'reconstruct', got {source!r}")
    if vectors.shape[0] != len(meta):
        raise RuntimeError(f"vector/metadata count mismatch: {vectors.shape[0]} vs {len(meta)}")

    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))
    if recreate or not client.collection_exists(collection):
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"(re)created collection {collection!r} (size={dim}, COSINE)")

    n, buf = 0, []
    for pid, vec, payload in build_points(vectors, meta):
        buf.append(PointStruct(id=pid, vector=vec, payload=payload))
        if len(buf) >= batch:
            client.upsert(collection_name=collection, points=buf)
            n += len(buf); buf = []
            print(f"  upserted {n}/{len(meta)}")
    if buf:
        client.upsert(collection_name=collection, points=buf); n += len(buf)

    count = client.count(collection_name=collection).count
    print(f"Done: source={source}, upserted {n}, collection now holds {count} points")
    return {"source": source, "upserted": n, "collection_count": count, "dim": dim}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-dir", type=Path, required=True,
                        help="dir with metadata.parquet + config.json (+ index.faiss)")
    parser.add_argument("--collection", required=True, help="target Qdrant collection name")
    parser.add_argument("--source", choices=["reembed", "reconstruct"], default="reembed",
                        help="vector source: reembed (default, exact) or reconstruct "
                             "(approximate IVFPQ fallback)")
    parser.add_argument("--batch", type=int, default=256, help="upsert batch size (default 256)")
    parser.add_argument("--recreate", action="store_true",
                        help="drop and recreate the collection first")
    args = parser.parse_args()
    migrate(args.index_dir, args.collection, source=args.source,
            batch=args.batch, recreate=args.recreate)


if __name__ == "__main__":
    main()
