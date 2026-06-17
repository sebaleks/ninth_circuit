# Confidence indicator — feature summary, deploy recipe, enabling gate

Branch `feat/confidence-indicator`. A per-result green/yellow/red indicator of how well a retrieved
case's TEXT matches the query (NOT a guarantee of legal relevance). Demo feature; OFF by default.

## Architecture
- **Two signals, 2-of-2 sum model** (`rag_api/confidence.py`): dense cosine + L-6 cross-encoder,
  each scored High/Med/Low (2/1/0) by calibrated cutoffs; sum ≥3 GREEN · 2 YELLOW · ≤1 RED. Robust
  by design — any single-signal error caps at YELLOW; RED needs both weak. Conservative low-end so a
  CE under-score can't push a real result to RED alone (the DV anchor stays GREEN).
- **CE fires only on the dense band [0.15, 0.50] of non-lookup queries** (`rag_api/cross_encoder.py`):
  dense<0.15 abstains (refused, no card), dense>0.50 serves on dense alone, docket/cite **lookups
  route around the CE** (it under-scores them) and are colored by retrieval strength (exact docket
  match → GREEN). Batched at 5 passages so peak RSS is bounded regardless of k.
- **Path B retrieval** (`rag_api/sparse_store.py` + retrieval.py): union-RRF over Qdrant dense +
  **Qdrant BM25 sparse** (off-box). META and the in-memory BM25 index are not loaded; BM25-only
  candidates are materialized from the dense collection's payloads. This frees the headroom the CE
  needs (see CONFIDENCE_MEMORY_FINDINGS.md).
- Model: **ms-marco-MiniLM-L-6-v2 @ top-5** (memory-driven choice; the plan's jina-tiny OOMs the
  512 MB box). 62.5% out_vocab catch.

## Six calibrated cutoffs (L-6-specific — do NOT transfer to another CE)
dense High≥0.33 / Med≥0.15 / Low<0.15 · CE High≥3.70 / Med≥0.80 / Low<0.80 · band [0.15, 0.50].

## Validation (this branch)
- **Memory:** real-app peak **484 MB < 512** (torch/faiss/rank_bm25/pandas all unloaded in Path B).
- **STEP 0 retriever gate:** Qdrant-BM25 reproduces/beats rank_bm25 (citation 0.907→0.987).
- **Gold (STEP 6):** thematic 21/22 green-top (the 1 red is a defensible weak #1 passage; the
  on-point passage in the same list is green); citation 48 green/27 yellow/0 red; authority 31/12/0.
  Zero refusals on in-corpus gold; CE fired on 86% of thematic, 0% of lookups (routing confirmed).
- **Latency:** retrieve ~0.6–1.0 s (network-dominated); CE adds ~120–240 ms on the band minority.
- 100 backend tests pass; default deployment (feature off) is byte-for-byte unchanged.

## Deploy recipe (to turn it ON)
Prereqs: the sparse collection `asylum_cases_nim2048_full_sparse` exists (built via
`evaluation/nim_e2e/build_full_sparse.py`); `fastembed==0.8.0` in requirements (added).
Env vars (in addition to the live VECTOR_STORE=qdrant / QDRANT_COLLECTION / INDEX_DIR):
```
BM25_BACKEND=qdrant
QDRANT_SPARSE_COLLECTION=asylum_cases_nim2048_full_sparse
CONFIDENCE_ENABLED=true
# already set: UNION_TOPK=100, RRF_K=60
```
Verify via `/health`: `bm25_backend=qdrant`, `confidence_enabled=true`. The L-6 CE + Qdrant/bm25 models
are **pre-baked into the Docker image** (`ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache` + a build-time
`RUN`, ~88 MB on disk) so there is **no first-query HuggingFace download** and no runtime HF dependency.
(Optional hardening: set `HF_HUB_OFFLINE=1` at runtime to forbid HF calls entirely — requires every CE/
BM25 model in use to be baked; don't set it globally if other paths fetch from HF.) Pre-baked weights
sit on disk and load into RAM only when the feature is on, so they don't affect the 484 MB peak.
Headroom is ~28 MB — fine for a single-user demo; thin for concurrent production.

## Enabling-for-real gate (before relying on it beyond the demo)
The 62.5% catch + the calibration rest on **105 mostly-synthetic adversaries** (fantasy queries:
vampire/werewolf/Wakanda). Before production reliance:
1. Build a **realistic out-of-corpus probe set** — plausible-but-wrong LEGAL queries (e.g. a real
   doctrine misapplied, an adjacent circuit's rule, a non-asylum immigration question), NOT fantasy.
2. Confirm **real protected-ground queries don't mislabel RED** on that set (the gang-resistance case
   shows real queries CAN go red when both signals are weak; quantify the rate on realistic data).
3. Re-confirm the **memory headroom on the Render Linux box** (these are macOS numbers) and under
   concurrency (each concurrent CE call spikes the arena).

## Future tuning (documented, not built)
v1 is symmetric (the sum model treats the two signals equally). Once the CE is validated on realistic
adversaries, consider **CE-weighting the disagreement cells**: make [Low-dense, High-CE] greener (the
CE rescues a passage dense under-pooled) and [High-dense, Low-CE] redder (the CE caught a vocab-match
the cosine couldn't) — i.e. let the more-trustworthy signal break ties asymmetrically instead of the
neutral YELLOW the sum model gives today.
