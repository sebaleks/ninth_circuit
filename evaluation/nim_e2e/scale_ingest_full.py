#!/usr/bin/env python3
"""Full-corpus ingest: ALL asylum_cases → page-aware chunk → NIM-2048 embed → Qdrant collection
asylum_cases_nim2048_full, plus the deploy artifacts (metadata.parquet + config.json) for INDEX_DIR.

Settled production config. RESUMABLE: reuses the cached 2k chunks; checkpoints the remaining
downloads to chunks_full_new.parquet; skips Qdrant point-ids already embedded. chunk_id is
re-indexed to the metadata-row position (production BM25↔dense fusion assumes that invariant).
Production e5 collection and the 2k collection are NEVER written.
"""
import os, sys, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = Path("/Users/sebastiansteen/Desktop/Asylum_RAG_Free")
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv; load_dotenv(str(ROOT / ".env"))
import pandas as pd, tiktoken
import pipeline.rag_ingest as ri
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from supabase import create_client

COLL = "asylum_cases_nim2048_full"
PROD = "asylum_cases_local_e5_384_onnx_clean"; TWOK = "asylum_cases_nim2048_2k"
CHUNKS_2K = ROOT / "evaluation/nim_e2e/chunks_2k.parquet"
NEW_CACHE = ROOT / "evaluation/nim_e2e/chunks_full_new.parquet"   # remaining cases (resume)
FULL_META = ROOT / "evaluation/nim_e2e/chunks_full.parquet"       # combined, chunk_id = row idx
INDEX_DIR = ROOT / "data/nim2048_full"                            # deploy dir
ATTR = ROOT / "evaluation/nim_e2e/attrition_full.json"
THIN_CHARS = 500; COLS = ["case_link","page","text","char_start","char_end","n_tokens","case_pub_status","case_disposition"]
ENC = tiktoken.get_encoding("cl100k_base")
client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"), timeout=120)
embed_client = ri.make_embed_client(); t0 = time.time()
assert COLL not in (PROD, TWOK), "guard: refuse to write production / 2k"

# ── Phase A: full case list from Supabase ──
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
rows_all = []
for start in range(0, 200000, 1000):
    r = sb.table("asylum_cases").select("link,published_status,final_disposition").range(start, start+999).execute().data
    if not r: break
    rows_all += r
cases = pd.DataFrame(rows_all).drop_duplicates("link")
print(f"Phase A: {len(cases)} cases from Supabase", flush=True)

# ── Phase B: reuse cached 2k chunks; download+chunk the rest (resumable) ──
cached = pd.read_parquet(CHUNKS_2K)[COLS]
have = set(cached["case_link"])
ok_buf = pd.read_parquet(NEW_CACHE).to_dict("records") if NEW_CACHE.exists() else []
done_new = {r["case_link"] for r in ok_buf}
attr = json.load(open(ATTR)) if ATTR.exists() else {"thin": [], "fail": []}
todo = [r for r in cases.to_dict("records") if r["link"] not in have and r["link"] not in done_new]
print(f"Phase B: cached2k={len(have)} cases, cached_new={len(done_new)} cases, to-fetch={len(todo)} cases", flush=True)

def process(row):
    link = row["link"]
    try:
        pages = ri.extract_pages(ri.download_pdf(link, timeout=60))
        if not pages or sum(len(p) for p in pages) < THIN_CHARS:
            return ("thin", link, [])
        out = []
        for pg, ptext in enumerate(pages, start=1):
            for ct, cs, ce, nt in ri.chunk_page(ptext, ENC):
                out.append({"case_link": link, "page": pg, "text": ct, "char_start": cs, "char_end": ce,
                            "n_tokens": nt, "case_pub_status": str(row.get("published_status", "")),
                            "case_disposition": str(row.get("final_disposition", ""))})
        return ("ok", link, out)
    except Exception as e:
        return ("fail", link, str(e)[:120])

if todo:
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(process, r) for r in todo]):
            st, link, payload = fut.result(); done += 1
            if st == "ok": ok_buf += payload
            elif st == "thin": attr["thin"].append(link)
            else: attr["fail"].append([link, payload])
            if done % 500 == 0:
                pd.DataFrame(ok_buf).to_parquet(NEW_CACHE, index=False)
                json.dump(attr, open(ATTR, "w"))
                print(f"  {done}/{len(todo)} | new_chunks={len(ok_buf)} thin={len(attr['thin'])} "
                      f"fail={len(attr['fail'])} ({(time.time()-t0)/60:.1f}m)", flush=True)
    pd.DataFrame(ok_buf).to_parquet(NEW_CACHE, index=False); json.dump(attr, open(ATTR, "w"))

# ── combine → full meta, chunk_id = row index ──
new_df = pd.DataFrame(ok_buf)[COLS] if ok_buf else pd.DataFrame(columns=COLS)
full = pd.concat([cached, new_df], ignore_index=True).reset_index(drop=True)
full["chunk_id"] = range(len(full))
full.to_parquet(FULL_META, index=False)
print(f"Combined: {len(full)} chunks across {full['case_link'].nunique()} cases", flush=True)

# ── Phase C: NIM-2048 embed + upsert (resumable) ──
if not client.collection_exists(COLL):
    client.create_collection(COLL, vectors_config=VectorParams(size=2048, distance=Distance.COSINE))
existing, off = set(), None
while True:
    pts, off = client.scroll(COLL, with_payload=False, with_vectors=False, limit=10000, offset=off)
    existing.update(p.id for p in pts)
    if off is None: break
todo_e = full[~full["chunk_id"].isin(existing)].reset_index(drop=True)
print(f"Phase C: {len(full)} chunks, {len(existing)} in Qdrant, {len(todo_e)} to embed", flush=True)

def embed_retry(texts, tries=7):
    for a in range(tries):
        try: return ri.embed_batch(embed_client, texts, input_type="passage", dim=2048)
        except Exception as e:
            w = min(90, 3 * 2 ** a); print(f"    NIM err try {a+1}: {str(e)[:80]} sleep {w}s", flush=True); time.sleep(w)
    raise RuntimeError("embed failed after retries")

def upsert_retry(points, tries=5):
    for a in range(tries):
        try:
            client.upsert(COLL, points=points); return
        except Exception as e:
            w = min(60, 4 * 2 ** a); print(f"    Qdrant upsert err try {a+1}: {str(e)[:80]} sleep {w}s", flush=True); time.sleep(w)
    raise RuntimeError("upsert failed after retries")

BATCH = ri.EMBED_BATCH; recs = todo_e.to_dict("records"); texts = todo_e["text"].tolist(); n_up = 0
for k in range(0, len(texts), BATCH):
    vb = embed_retry(texts[k:k+BATCH]); rb = recs[k:k+BATCH]
    upsert_retry([PointStruct(id=int(r["chunk_id"]), vector=vb[i].tolist(),
        payload={"chunk_id": int(r["chunk_id"]), "case_link": r["case_link"], "snippet": r["text"],
                 "page": int(r["page"]), "case_pub_status": r["case_pub_status"],
                 "case_disposition": r["case_disposition"]}) for i, r in enumerate(rb)])
    n_up += len(vb)
    if (k // BATCH) % 20 == 0:
        print(f"  embedded {n_up}/{len(todo_e)} ({(time.time()-t0)/60:.1f}m)", flush=True)

# ── Phase D: deploy artifacts + report ──
INDEX_DIR.mkdir(parents=True, exist_ok=True)
full.to_parquet(INDEX_DIR / "metadata.parquet", index=False)
json.dump({"embedder": "nim", "model_id": "nvidia/llama-nemotron-embed-1b-v2", "dim": 2048,
           "chunk_size": 1500, "overlap": 150}, open(INDEX_DIR / "config.json", "w"), indent=2)
cnt = client.count(COLL).count; ncase = int(full["case_link"].nunique())
rep = {"collection": COLL, "supabase_cases": len(cases), "usable_cases": ncase, "chunks": int(cnt),
       "chunks_per_case": round(cnt / ncase, 2), "storage_gb": round(cnt * 2048 * 4 * 1.3 / 1e9, 3),
       "thin": len(attr["thin"]), "fail": len(attr["fail"]), "wall_min": round((time.time()-t0)/60, 1),
       "index_dir": str(INDEX_DIR.relative_to(ROOT)), "qdrant_collection": COLL}
json.dump(rep, open(ROOT / "evaluation/nim_e2e/results/scale_ingest_full.json", "w"), indent=2)
print("=" * 60); print(f"DONE → {COLL}"); print(json.dumps(rep, indent=2)); print("=" * 60, flush=True)
