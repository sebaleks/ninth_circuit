#!/usr/bin/env python3
"""STEP 6: validate the confidence indicator on the VERIFIED GOLD via the real Path B pipeline.

Runs every gold query (thematic / citation / named-authority) through union-RRF (Qdrant dense +
Qdrant BM25) + the L-6 CE annotator, and reports the color distribution (does real in-corpus skew
green/yellow, never red?), the % of queries that actually invoke the CE (the band non-lookup
minority), and latency. Offline; reads live collections; no writes except the JSON report.
"""
import os, sys, json, time
from pathlib import Path
from collections import Counter
ROOT = Path("/Users/sebastiansteen/Desktop/Asylum_RAG_Free"); os.chdir(ROOT); sys.path.insert(0, str(ROOT))
os.environ.update(VECTOR_STORE="qdrant", QDRANT_COLLECTION="asylum_cases_nim2048_full",
                  QDRANT_SPARSE_COLLECTION="asylum_cases_nim2048_full_sparse",
                  BM25_BACKEND="qdrant", INDEX_DIR="data/nim2048_full", CONFIDENCE_ENABLED="true")
from dotenv import load_dotenv; load_dotenv(str(ROOT / ".env"))
from rag_api import retrieval, cross_encoder, guardrails, confidence as conf
retrieval.load()
assert retrieval.META is None and retrieval.SPARSE_STORE is not None, "expected Path B"

def gold(name):
    d = json.load(open(ROOT / f"evaluation/nim_e2e/results/{name}"))
    return d["queries"] if isinstance(d, dict) else d
SETS = {"thematic": gold("thematic_gold_2k.json"),
        "citation": gold("citation_gold_2k.json"),
        "named_authority": gold("named_authority_gold_2k.json")}

report = {}
for typ, qs in SETS.items():
    top_colors = Counter(); any_red = []; ce_fired = 0; refused = 0; lat = []
    for q in qs:
        query = q["query"]
        t = time.time()
        hits = retrieval.search_with_rerank(query, fetch_k=20, return_k=5)
        if guardrails.should_refuse([h.get("dense_score", h["score"]) for h in hits]):
            refused += 1; lat.append((time.time()-t)*1000); continue
        cross_encoder.annotate(query, hits)
        lat.append((time.time()-t)*1000)
        if any(h.get("ce_score") is not None for h in hits): ce_fired += 1
        if hits:
            top_colors[hits[0]["confidence"]["color"]] += 1
            if hits[0]["confidence"]["color"] == "red":
                any_red.append((query[:50], round(hits[0]["dense_score"], 3)))
    n = len(qs); slat = sorted(lat)
    report[typ] = {"n": n, "top_color": dict(top_colors), "refused": refused,
                   "ce_fired_pct": round(100*ce_fired/n, 1),
                   "median_latency_ms": round(slat[len(slat)//2]) if slat else 0,
                   "top_red_examples": any_red[:5]}
    print(f"\n{typ} (n={n}):")
    print(f"  top-result color: {dict(top_colors)}  | refused: {refused}")
    print(f"  CE fired on {report[typ]['ce_fired_pct']}% of queries | median latency {report[typ]['median_latency_ms']}ms")
    if any_red: print(f"  TOP-RESULT RED ({len(any_red)}): {any_red[:5]}")

json.dump(report, open(ROOT / "evaluation/nim_e2e/results/confidence_gold_validation.json", "w"), indent=2)
print("\nwrote confidence_gold_validation.json")
