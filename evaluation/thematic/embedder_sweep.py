#!/usr/bin/env python3
"""Embedder sweep — score candidate embedders on (A) the 6-query retrieval gold and
(B) FGM/off-domain abstention separation, holding the pipeline fixed (RRF + β=0
case-collapse + dense-gated pool) so only the embedder varies.

Configs: 4 local 384-dim ST models + NIM nemotron Matryoshka @ 512/1024/2048
(embedded once at 2048, truncated+renormalized — the Matryoshka contract).

Writes a markdown report to evaluation/thematic/embedder_sweep_results.md.
"""
from __future__ import annotations
import os, re, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# NIM key (authorized; never printed)
if not os.environ.get("NVIDIA_API_KEY"):
    for line in (REPO/".env").read_text().splitlines():
        if line.strip().startswith("NVIDIA_API_KEY="):
            os.environ["NVIDIA_API_KEY"] = line.split("=",1)[1].strip().strip('"').strip("'")

STOP={"the","and","for","with","from","that","this","what","where","when","which","who","whom",
      "does","did","was","were","are","have","has","had","but","not","all","any","some","into","about","case","cases"}
qtok=lambda s:[t for t in re.findall(r"[a-z0-9]+",s.lower()) if len(t)>=3 and t not in STOP]
docket=lambda l: (re.search(r"/(\d{2}-\d+)\.pdf",l or "") or [None,"?"])[1] if l else "?"
def dk(l):
    m=re.search(r"/(\d{2}-\d+)\.pdf",l or ""); return m.group(1) if m else "?"

df=pd.read_parquet(REPO/"data/experiments/T-optimized-onnx-clean/metadata.parquet")
texts=df.text.astype(str).tolist(); links=df.case_link.tolist()
bm25=BM25Okapi([qtok(t) for t in texts]); RRF_K=60

suite=json.loads((REPO/"evaluation/thematic/thematic_queries.json").read_text())
GOLD=[(q["id"], q["query"], [dk(l) for l in q["gold"]["core"]], q.get("sentinel")) for q in suite["queries"]]
ABST={"in":["sexual assault","particular social group","gang violence"],
      "absent":["female genital cutting","female genital mutilation","forced recruitment into a militia"],
      "off":["capital of France","weather in Boston today"]}

def rrf_topk(D, qv, k=5):
    dc=D@qv; pool=list(np.argsort(-dc)[:max(20,k*5)])
    return dc, pool

def top_cases(D, qv, k=5):
    dc=D@qv; pool=list(np.argsort(-dc)[:max(20,k*5)])
    # bm25 norm corpus-wide, ranked within pool (dense-gated) — matches serving
    dr={i:r for r,i in enumerate(sorted(pool,key=lambda i:-dc[i]),1)}
    br={i:r for r,i in enumerate(sorted(pool,key=lambda i:-bm25_norm_cache[i]),1)}
    fused=sorted(pool,key=lambda i:-(1/(RRF_K+dr[i])+1/(RRF_K+br[i])))
    seen,out=set(),[]
    for i in fused:
        if links[i] in seen: continue
        seen.add(links[i]); out.append(dk(links[i]))
        if len(out)>=k: break
    return out

# ── embedder backends ────────────────────────────────────────────────────────
def local_backend(model_id, qp, pp, trust=False):
    from sentence_transformers import SentenceTransformer
    m=SentenceTransformer(model_id, trust_remote_code=trust)
    D=m.encode([pp+t for t in texts],normalize_embeddings=True,show_progress_bar=False).astype(np.float32)
    return D, (lambda q: m.encode(qp+q,normalize_embeddings=True).astype(np.float32))

# approx params + int8 ONNX size + whether it fits the 512 MB Render free tier
FIT={"e5-small-v2 (current)":"33M / ~35MB / ✅","e5-base-v2":"109M / ~110MB / ✅",
     "e5-large-v2":"335M / ~335MB / ◑ tight","gte-base":"109M / ~110MB / ✅",
     "bge-base-en-v1.5":"109M / ~110MB / ✅","nomic-embed-text-v1.5":"137M / ~140MB / ✅"}

_NIM_CACHE={}
def nim_2048():
    if "D" not in _NIM_CACHE:
        from rag_api import nvidia_client
        parts=[]
        for i in range(0,len(texts),64):
            parts.append(np.asarray(nvidia_client.embed_passages(texts[i:i+64]),dtype=np.float32))
            print(f"   NIM corpus {min(i+64,len(texts))}/{len(texts)}",flush=True)
        _NIM_CACHE["D"]=np.vstack(parts)
        _NIM_CACHE["qf"]=lambda q: np.asarray(nvidia_client.embed_query(q),dtype=np.float32).reshape(-1)
    return _NIM_CACHE["D"], _NIM_CACHE["qf"]

def trunc(v,d):
    v=v[...,:d]; n=np.linalg.norm(v,axis=-1,keepdims=True); return (v/np.where(n==0,1,n)).astype(np.float32)

def nim_backend(d):
    D2,qf2=nim_2048()
    D=trunc(D2,d)
    return D, (lambda q: trunc(qf2(q),d))

CONFIGS=[
 ("e5-small-v2 (current)", lambda: local_backend("intfloat/e5-small-v2","query: ","passage: ")),
 ("e5-base-v2",            lambda: local_backend("intfloat/e5-base-v2","query: ","passage: ")),
 ("e5-large-v2",           lambda: local_backend("intfloat/e5-large-v2","query: ","passage: ")),
 ("gte-base",              lambda: local_backend("thenlper/gte-base","","")),
 ("bge-base-en-v1.5",      lambda: local_backend("BAAI/bge-base-en-v1.5","Represent this sentence for searching relevant passages: ","")),
 ("nomic-embed-text-v1.5", lambda: local_backend("nomic-ai/nomic-embed-text-v1.5","search_query: ","search_document: ",True)),
]

rows=[]
for name,build in CONFIGS:
    print(f"\n=== {name} ===",flush=True)
    try:
        D,qf=build(); dim=int(D.shape[1])
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}",flush=True); rows.append({"name":name,"dim":"?","err":str(e)}); continue
    global bm25_norm_cache
    # retrieval gold
    gold_detail={}; core_hit=0; core_tot=0
    for qid,q,core,sent in GOLD:
        qv=qf(q)
        raw=bm25.get_scores(qtok(q)); mx=raw.max(); bm25_norm_cache=(raw/mx) if mx>0 else raw*0
        top=top_cases(D,qv)
        ranks={c:(top.index(c)+1 if c in top else None) for c in core}
        core_hit+=sum(1 for r in ranks.values() if r); core_tot+=len(core)
        gold_detail[qid]={"top5":top,"core_ranks":ranks}
    # abstention separation
    def mc(q): return float((D@qf(q)).max())
    inv=[mc(q) for q in ABST["in"]]; ab=[mc(q) for q in ABST["absent"]]; of=[mc(q) for q in ABST["off"]]
    gap=min(inv)-max(ab+of)
    rows.append({"name":name,"dim":dim,"core_hit":core_hit,"core_tot":core_tot,
                 "gold":gold_detail,"in":inv,"absent":ab,"off":of,"gap":gap})
    print(f"   gold core recovered {core_hit}/{core_tot}  | abstention gap={gap:+.3f} (in_min {min(inv):.3f} vs absent/off_max {max(ab+of):.3f})",flush=True)

# ── markdown report ────────────────────────────────────────────────────────
def fmt_rank(r): return str(r) if r else "—"
out=["# Embedder sweep (mid-size local models) — 6-query gold + abstention\n",
     "Pipeline fixed (RRF + β=0 case-collapse + dense-gated pool); only the embedder varies. "
     "Local models 384-dim; NIM via Matryoshka truncation from 2048.\n",
     "## A. Retrieval quality (core case recovered in top-5; rank shown)\n",
     "| embedder | dim | SA 14-70905 | DV 23-4420 | POL 24-2787 | POL 20-72806 | vocab 25-120 | PSG 17-72197 | CITE 21-70493 | core rec |",
     "|---|---|---|---|---|---|---|---|---|---|"]
def g(r,qid,d):
    return fmt_rank(r["gold"].get(qid,{}).get("core_ranks",{}).get(d))
for r in rows:
    if r.get("err"): out.append(f"| {r['name']} | {r['dim']} | ERROR: {r['err'][:40]} |||||||"); continue
    out.append(f"| {r['name']} | {r['dim']} | {g(r,'A-sexual-assault','14-70905')} | {g(r,'A-domestic-violence','23-4420')} | "
               f"{g(r,'A-political-imprisonment','24-2787')} | {g(r,'A-political-imprisonment','20-72806')} | "
               f"{g(r,'B-vocab-mismatch','25-120')} | {g(r,'B-doctrinal-psg','17-72197')} | {g(r,'C-citation-lat01','21-70493')} | "
               f"**{r['core_hit']}/{r['core_tot']}** |")
out+=["\n## B. Abstention separation (dense max-cosine; gap = in-corpus min − max(absent,off))\n",
      "| embedder | dim | in-corpus min | FGM/absent max | off-domain max | gap | separable? | params/int8/512MB |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    if r.get("err"): continue
    sep = "✓ YES" if r["gap"]>0.02 else ("~ thin" if r["gap"]>0 else "✗ NO")
    out.append(f"| {r['name']} | {r['dim']} | {min(r['in']):.3f} | {max(r['absent']):.3f} | {max(r['off']):.3f} | {r['gap']:+.3f} | {sep} | {FIT.get(r['name'],'—')} |")
report="\n".join(out)+"\n"
(REPO/"evaluation/thematic/embedder_sweep_midsize_results.md").write_text(report)
print("\n"+report)
print("WROTE evaluation/thematic/embedder_sweep_midsize_results.md")
