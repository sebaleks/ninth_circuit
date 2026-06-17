# Architecture Decisions — Retrieval

Settled retrieval configuration for the Ninth-Circuit asylum RAG, wired into
[`rag_api/retrieval.py`](rag_api/retrieval.py). Each decision lists its one-line evidence from
the experiment series (`evaluation/nim_e2e/results/`). **Preliminary** — validated on the
applied-but-unreviewed gold; finalize after the gold-set review.

---

## Settled config

| Layer | Decision | One-line evidence |
|---|---|---|
| **Dense** | NIM `nvidia/llama-nemotron-embed-1b-v2` @ **2048 dims**, page-aware chunking (token-1500/page) | Fits free storage at **~0.33 GB** full-corpus projection (the "1.46 GB" was a biased 30-case pilot artifact). |
| **Lexical** | **BM25 as a FULL retriever** (its own top-k), in-memory, **`sys.intern`** token-interning | Always-on BM25 footprint **76 MB → 31 MB (−59%)** @10k → **~94 MB at full ~31k corpus** → fits the 512 MB box. |
| **Fusion** | **UNION-RRF** — dense and BM25 each retrieve top-`UNION_TOPK`, fused by reciprocal rank (`RRF_K`) | Beats the in-pool blend on lookups (the blend is capped by dense's poor pool coverage); ties dense on thematic. |
| **Reranker** | **OFF** | Cross-encoder gave noise-level gain (**MRR +0.02**) for **~6.7× server latency** (225 → 1,510 ms). Dropped. |

### Why union-RRF over the in-pool blend (the load-bearing choice)
The old hybrid **re-weighted the dense pool** with BM25, so it could only reorder what dense
already retrieved. But dense pools only **~39% of docket** cases and **~20% of named-authority**
cases in its top-100 — it can't embed opaque tokens (docket numbers, surnames, statute cites).
Union-RRF lets BM25 contribute its **own** candidates, so it *rescues* exactly what dense misses:

| query type | dense-only | in-pool blend (old) | **union-RRF (settled)** |
|---|---|---|---|
| thematic (recall@10) | 0.521 | 0.531 | **0.526** (no regression) |
| citation / docket | 0.227 | 0.387 | **0.960** |
| named-authority | 0.067 | 0.100 | **0.208** |

Dense ≈ union on thematic (BM25 adds nothing to conceptual queries — and slightly helps via the
exact-match path), but dense **collapses** on the two lookup classes where union-RRF recovers it.
Exact-match value is **broader than bare dockets** (it covers case names + statutes too), which is
why a docket-regex bypass is insufficient and BM25/lexical stays in the path.

---

## Config knobs (env; defaults are the settled values)

| env var | default | meaning |
|---|---|---|
| `FUSION_METHOD` | `union_rrf` | retrieval fusion; `blend`/`rrf` = legacy in-pool modes (ablation only) |
| `UNION_TOPK` | `100` | per-retriever depth — dense AND BM25 each fetch this many before fusing |
| `RRF_K` | `60` | reciprocal-rank-fusion constant |
| `USE_RERANKER` | `true` | **not used by union-RRF** (the reranker is bypassed in this fusion) |
| `VECTOR_STORE` | `faiss` | dense backend (`faiss` / `qdrant`) |

---

## PARKED — Option 1: query-class router (deferred, not rejected)

A router would classify each query — *lookup* (docket / case-name / statute) → lexical-first;
*conceptual* → dense-only — running **one** retriever per query (leaner than always-on union).

**Why parked:** it needs a reliable lookup-vs-conceptual classifier, and the named-authority test
proved lookups are **open-ended tokens with no clean regex** — a misclassified lookup silently
loses its exact-match path. Always-on union-RRF was chosen first: **no classifier**, negligible
thematic cost (recall flat, MRR −0.05, +0.63 ms fusion). **Revisit if** always-on BM25's
memory/latency tightens at much larger scale, **or** a high-precision cite-token classifier is built.

The retrieval entry point (`search_with_rerank`) is structured so the router can wrap it without
touching the fusion internals — see the **ROUTER SEAM** comment there.

---

## Validation (preliminary, current gold)

- Union-RRF reproduces the lifts: thematic **0.526** (no regression vs dense 0.521), citation
  **0.960**, named-authority **0.208** — see [`unionrrf_validation.json`](evaluation/nim_e2e/results/unionrrf_validation.json).
- BM25 memory post-`sys.intern`: **31.2 MB @10k → ~94 MB projected at full corpus** (fits 512 MB).
- Union-RRF fusion step **~0.63 ms/query**; dense Qdrant query dominates and is unchanged.
- Full test suite green (93 passed).
