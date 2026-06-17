# Confidence indicator — STEP 0 (retriever) + memory gate findings

Branch `feat/confidence-indicator`. Both pre-build gates run before wiring the feature.

## STEP 0 — Qdrant-BM25 lookup re-validation (GATE: PASS)
Moving BM25 off-box (Path B) swaps in-memory rank_bm25 (no stemming) for Qdrant fastembed
`Qdrant/bm25` (snowball stemming + IDF) — a *different* retriever (prior top-10 agreement only
0.476). Re-ran union-RRF on the **full corpus** with both BM25 backends side-by-side (same dense,
same corpus → isolates the retriever). recall@10 / MRR / NDCG:

| query type | dense-only | union rank_bm25 | union **Qdrant-BM25** |
|---|---|---|---|
| citation (n=75) | 0.133/0.083/0.092 | 0.907/0.451/0.560 | **0.987/0.520/0.637** |
| named-authority (n=43) | 0.044/0.300/0.150 | 0.106/0.425/0.239 | **0.110**/0.443/0.253 |
| thematic (n=22) | 0.245/0.567/0.349 | 0.278/0.610/0.377 | 0.278/0.537/0.359 |

**Verdict: Qdrant-BM25 reproduces or beats rank_bm25 on all three.** Citation 0.907→**0.987**
(stemming+IDF helps dockets), authority ≈ (0.106→0.110), thematic identical recall (not regressed
vs dense). Caveat: authority/thematic absolute numbers are lower than the prior **2k-corpus** numbers
(0.21, 0.526) — that's the 2k-gold-on-full-corpus distractor effect (both retrievers drop equally),
NOT the retriever. Citation (1 exact case) is corpus-size-robust → **0.987 is the lookup headline**.
Full sparse collection built: `asylum_cases_nim2048_full_sparse`, 30,021 points (chunk_id == dense id).

## Memory gate — CE on the 512 MB Render free tier
Peak RSS (`getrusage` maxrss), torch-free (onnxruntime path, representative of deploy). All configs:
META off-box (snippets from Qdrant payloads), in-memory BM25 off-box (Path B).

**Naive stack is over budget** — and faiss is the hidden culprit:
| stack | peak |
|---|---|
| keep META + jina-tiny CE | 715 MB |
| no META, full app imports (incl. faiss) + L-6, warmed | 522 MB |
| **trim faiss + rank_bm25 imports** (unused in qdrant Path B) + L-6 | **379 MB** (idle) |

faiss (BLAS/OpenMP) + onnxruntime together inflate the high-water mark; faiss is imported at
`retrieval.py` top but UNUSED in qdrant mode. Trimming it is the unlock.

**Realistic per-query load** (band query scores its top-k retrieved passages ~300 words, max-pooled,
one batch/query; threads=1, `enable_cpu_mem_arena=False`, faiss/bm25 trimmed):
| model | nonsense-catch | top-5 peak | top-3 peak | fits 512 MB |
|---|---|---|---|---|
| **L-6** (ms-marco-MiniLM-L-6-v2) | 62.5% | **475 MB** | — | ✓ (validated top-5) |
| L-12 (ms-marco-MiniLM-L-12-v2) | 71.9% | 531 MB | **485 MB** | ✓ only at top-3 |
| jina-reranker-v1-tiny-en | 71.9% | 558 MB | — | ✗ |

**Key deviation from the plan:** the plan named **jina-tiny** ("fits 512 MB once BM25→Qdrant"). It
does NOT (558 MB). Memory inverts the eval's latency-based pick: the ms-marco models are lighter, and
**L-12 ties jina-tiny's 71.9% catch at less memory**. The deployable free-tier choices are **L-6 @
top-5 (475 MB, validated, 62.5%)** or **L-12 @ top-3 (485 MB, 71.9%, batch needs re-validation)**.

**Required to fit:** (1) lazy/conditional faiss+rank_bm25 imports in qdrant mode, (2) META off-box
(snippets from Qdrant), (3) onnxruntime threads=1 + arena off, (4) CE batch capped (top-k passages).
Headroom is ~30–40 MB — fine for a single-user demo; thin for concurrent production (the enabling-
for-real gate). Bigger instance (Render Standard 2 GB) removes all of this and allows any model @ top-5.

## Calibration (L-6 @ top-5) — the six cutoffs + anchors
| signal | High ≥ | Med ≥ (M/L) | Low < | basis |
|---|---|---|---|---|
| dense cosine | **0.33** | **0.15** | 0.15 | in-corpus p25=0.325; off-topic ≤0.147 (abstention floor) |
| CE (L-6) | **3.70** | **0.80** | 0.80 | in-corpus in-band p25=3.70; M/L=0.80 < DV anchor 1.00 (conservative low-end) |

Band [0.15, 0.50]; docket/cite lookups route around the CE. Color = 2-of-2 sum (≥3 green / 2 yellow / ≤1 red).

**Anchor cases:** real-strong (d0.50/ce5.7)→GREEN · **DV under-scored (d0.462/ce1.00)→GREEN** (ce≥0.80=Med
+ dense High = sum 3, protected) · out_vocab caught (d0.43/ce−1.5)→YELLOW · weak-dense nonsense
(d0.28/ce−1.5)→RED · off-topic (d<0.15)→abstained.

**Validation on the probe set (color applied to every probe):** in_thematic 16/16 GREEN · in_edge 15/16
green/yellow (the 1 "red" is `MPP standing` d0.117 — abstained below the floor, never shown) · in_lookup
27/27 green/yellow · out_clear 14/14 RED · out_vocab **17/32 (53%) pulled to yellow/red**; the 15 that stay
green are the elaborate disguised-fiction adversaries (vampire/merfolk/Wakanda/mock-opinion language) —
L-6's known ceiling. **No real in-corpus query displays RED.** Under-warn over over-warn, as specified.

## Path B implementation — real-app memory + latency (measured)
Built: BM25→Qdrant sparse (`SparseBM25Store`), META off-box (BM25-only union candidates
materialized from the dense collection's payloads via `QdrantStore.retrieve_payloads`), and
lazy faiss/rank_bm25/**pandas** imports (all unused in qdrant Path B). Staged peak RSS of the
REAL app (qdrant dense + qdrant BM25 + NIM embed client + L-6 CE @ top-5, CONFIDENCE_ENABLED):

| stage | peak RSS |
|---|---|
| import retrieval (lazy pandas) | 71 MB |
| load() Path B (META=None, sparse store) | 133 MB |
| + 1 dense+sparse query | 148 MB |
| **+ CE annotate (model load + inference)** | **484 MB** |

**484 MB < 512** (28 MB headroom). Confirmed unloaded: torch, faiss, rank_bm25, transformers,
pandas. Without the lazy-pandas trim it was 541 MB (over). Correctness verified end-to-end: the
exact docket `23-2038` (dense missed it) is recovered by Qdrant BM25 and its snippet materialized
from the dense payload → GREEN.

**Latency correction:** Path B does NOT reduce end-to-end latency. Both paths are network-dominated
(NIM embed + free-tier Qdrant ≈ 1 s); Path B actually ADDS a sparse query + a payload fetch (~300 ms)
versus the local in-memory `BM25.get_scores` (the in-memory index builds ONCE at startup, cached, so
it's fast per-query). **Path B's real justification is MEMORY — it's what frees the headroom for the
CE — not latency.** CE adds ~120–240 ms on the band, non-lookup minority of queries only.
