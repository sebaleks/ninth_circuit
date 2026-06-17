#!/usr/bin/env python3
"""STEP 0 gate: does Qdrant-BM25 (fastembed, snowball-stemmed) reproduce the union-RRF lookup
recall that in-memory rank_bm25 produced? Runs union-RRF on the FULL corpus with BOTH BM25
backends side-by-side (same dense, same corpus -> isolates the retriever), + dense-only, across
citation / named-authority / thematic gold. Compares to the prior rank_bm25 numbers.
Offline; reads the live dense + sparse collections; no writes.
"""
import os, sys, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from dotenv import load_dotenv
ROOT = Path("/Users/sebastiansteen/Desktop/Asylum_RAG_Free"); os.chdir(ROOT); sys.path.insert(0, str(ROOT))
load_dotenv(str(ROOT / ".env"))
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from fastembed import SparseTextEmbedding
import rag_api.nvidia_client as nim
from rag_api.retrieval import _query_tokens
DENSE = "asylum_cases_nim2048_full"; SPARSE = "asylum_cases_nim2048_full_sparse"
POOL = 100; RRF_K = 60
client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"), timeout=120)
m = pd.read_parquet(ROOT / "data/nim2048_full/metadata.parquet")[["chunk_id", "case_link", "text"]]
cids = m["chunk_id"].tolist(); links = m["case_link"].tolist(); cid2row = {c: i for i, c in enumerate(cids)}
print(f"building in-memory rank_bm25 over {len(m)} chunks...", flush=True)
bm25 = BM25Okapi([_query_tokens(t) for t in m["text"].tolist()])
spmodel = SparseTextEmbedding(model_name="Qdrant/bm25")

TH = json.load(open(ROOT / "evaluation/nim_e2e/results/thematic_gold_2k.json"))["queries"]
CG = json.load(open(ROOT / "evaluation/nim_e2e/results/citation_gold_2k.json")); CG = CG["queries"] if isinstance(CG, dict) else CG
NA = json.load(open(ROOT / "evaluation/nim_e2e/results/named_authority_gold_2k.json"))["queries"]
SETS = {"thematic": TH, "citation": CG, "named_authority": NA}

def dedupe(pairs):
    seen, out = set(), []
    for lk, _ in pairs:
        if lk not in seen: seen.add(lk); out.append(lk)
    return out
def rrf(rankings):
    sc = defaultdict(float)
    for rk in rankings:
        for r, cid in enumerate(rk, 1): sc[cid] += 1.0 / (RRF_K + r)
    return [c for c, _ in sorted(sc.items(), key=lambda x: -x[1])]
def ev(cases, gold):
    core = set(gold["core"]); marg = set(gold.get("marginal", {})); top = cases[:10]
    rec = len([c for c in top if c in core]) / max(1, len(core))
    mrr = next((1.0 / i for i, c in enumerate(cases, 1) if c in core), 0.0)
    g = lambda c: 2 if c in core else (1 if c in marg else 0)
    dcg = sum(g(c) / math.log2(i + 1) for i, c in enumerate(top, 1))
    idcg = sum(v / math.log2(i + 1) for i, v in enumerate(sorted([2]*len(core)+[1]*len(marg), reverse=True)[:10], 1))
    return rec, mrr, (dcg / idcg if idcg > 0 else 0.0)

def dense_pool(qtext):
    v = nim.embed_query(qtext, dim=2048)[0].tolist()
    pts = client.query_points(DENSE, query=v, limit=POOL, with_payload=True).points
    return [(p.payload["chunk_id"], p.payload["case_link"]) for p in pts]
def rank_bm25_top(qtext):
    raw = bm25.get_scores(_query_tokens(qtext)); order = np.argsort(-raw)[:POOL]
    return [cids[i] for i in order if raw[i] > 0]
def qdrant_bm25_top(qtext):
    qv = list(spmodel.query_embed([qtext]))[0]
    pts = client.query_points(SPARSE, query=models.SparseVector(indices=qv.indices.tolist(), values=qv.values.tolist()),
                              using="bm25", limit=POOL, with_payload=True).points
    return [p.payload["chunk_id"] for p in pts]

R = {c: defaultdict(list) for c in ("dense_only", "union_rank_bm25", "union_qdrant_bm25")}
n = sum(len(v) for v in SETS.values())
print(f"validating {n} queries (dense embed each) ...", flush=True)
for typ, qs in SETS.items():
    for q in qs:
        pool = dense_pool(q["query"]); dense_cids = [c for c, _ in pool]; link = {c: lk for c, lk in pool}
        rb = rank_bm25_top(q["query"]); qb = qdrant_bm25_top(q["query"])
        def to_cases(cidorder):
            return dedupe([(link.get(c) or links[cid2row[c]], 0) for c in cidorder])
        d_cases = dedupe([(lk, 0) for _, lk in pool])
        u_rb = to_cases(rrf([dense_cids, rb]))
        u_qb = to_cases(rrf([dense_cids, qb]))
        for name, cs in [("dense_only", d_cases), ("union_rank_bm25", u_rb), ("union_qdrant_bm25", u_qb)]:
            r, mr, nd = ev(cs, q["gold"]); R[name][typ + "|recall"].append(r); R[name][typ + "|mrr"].append(mr); R[name][typ + "|ndcg"].append(nd)

out = {"corpus": DENSE, "pool": POOL, "prior_rank_bm25_2k": {"citation_recall": 0.96, "named_authority_recall": 0.21, "thematic_recall": 0.526}, "config": {}}
for name in R:
    out["config"][name] = {t: {k.split("|")[1]: round(float(np.mean(v)), 4) for k, v in R[name].items() if k.startswith(t)} for t in SETS}
json.dump(out, open(ROOT / "evaluation/nim_e2e/results/qdrantbm25_validation.json", "w"), indent=2)
print("\n=== union-RRF recall@10 / MRR / NDCG on the FULL corpus (retriever isolated) ===")
for typ in SETS:
    print(f"\n{typ} (n={len(SETS[typ])}):")
    for name in ("dense_only", "union_rank_bm25", "union_qdrant_bm25"):
        mt = out["config"][name][typ]; print(f"  {name:<20} {mt['recall']:.3f} / {mt['mrr']:.3f} / {mt['ndcg']:.3f}")
print("\nprior rank_bm25 (2k corpus): citation 0.96 / authority 0.21 / thematic 0.526")
