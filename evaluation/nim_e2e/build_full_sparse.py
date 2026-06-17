#!/usr/bin/env python3
"""Build the full-corpus Qdrant BM25 SPARSE collection (off-box, persisted) for Path B.
fastembed Qdrant/bm25 (snowball stemming) + IDF modifier, over the 30,021 chunks; point id =
chunk_id (== metadata row index), so it fuses with the dense collection by chunk_id. Resumable;
upsert retry. Does NOT touch the live dense collection or the verified gold.
"""
import os, sys, time
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path("/Users/sebastiansteen/Desktop/Asylum_RAG_Free"); os.chdir(ROOT)
load_dotenv(str(ROOT / ".env"))
import pandas as pd
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
SRC = "asylum_cases_nim2048_full"; SP = "asylum_cases_nim2048_full_sparse"
assert SP.endswith("_sparse")
client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"), timeout=120)
m = pd.read_parquet(ROOT / "data/nim2048_full/metadata.parquet")[["chunk_id", "case_link", "text"]]
print(f"{len(m)} chunks to sparse-index", flush=True)

if client.collection_exists(SP):
    existing, off = set(), None
    while True:
        pts, off = client.scroll(SP, limit=10000, with_payload=False, offset=off)
        existing.update(p.id for p in pts)
        if off is None: break
    print(f"[resume] {len(existing)} sparse points already present", flush=True)
else:
    client.create_collection(SP, vectors_config={},
                             sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    existing = set()
    print(f"created {SP} (sparse-only, bm25/IDF)", flush=True)

todo = m[~m["chunk_id"].isin(existing)].reset_index(drop=True)
print(f"{len(todo)} to embed+upsert", flush=True)
model = SparseTextEmbedding(model_name="Qdrant/bm25")
recs = todo.to_dict("records"); t0 = time.time()
embs = model.embed(todo["text"].tolist(), batch_size=256)

def upsert_retry(pts, tries=5):
    for a in range(tries):
        try:
            client.upsert(SP, points=pts); return
        except Exception as e:
            w = min(60, 4 * 2 ** a); print(f"  upsert retry {a+1}: {str(e)[:70]} sleep {w}s", flush=True); time.sleep(w)
    raise RuntimeError("upsert failed after retries")

B, buf, n = 1000, [], 0
for r, e in zip(recs, embs):
    buf.append(models.PointStruct(id=int(r["chunk_id"]),
        vector={"bm25": models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())},
        payload={"chunk_id": int(r["chunk_id"]), "case_link": r["case_link"]}))
    if len(buf) >= B:
        upsert_retry(buf); n += len(buf); buf = []
        if n % 5000 == 0: print(f"  upserted {n}/{len(recs)} ({(time.time()-t0)/60:.1f}m)", flush=True)
if buf: upsert_retry(buf); n += len(buf)
cnt = client.count(SP).count
print(f"DONE: {cnt} sparse points in {SP} ({(time.time()-t0)/60:.1f}m)", flush=True)
