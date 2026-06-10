# Retrieval-quality investigation — findings & reasoning

> **Status (2026-06-09): validated fixes LIVE.** RRF + case-collapse (β=0) + content
> hygiene shipped to prod (commit `c27fffd`, collection
> `asylum_cases_local_e5_384_onnx_clean`); `/health` + smoke tests confirm the live
> serving path reproduces the sim byte-for-byte. gte and β>0 rejected. Details: §8.

**Scope:** Why the Ninth-Circuit asylum RAG buries genuinely-relevant cases on harm
queries (e.g. "sexual assault") and exact-citation lookups (lat-01), what is and
isn't the cause, and what to ship. Corpus: 30 published opinions, 687 chunks,
local e5-small-v2 (ONNX) + Qdrant, hybrid 0.6 dense / 0.4 BM25.

**Traceability:** every claim below cites an artifact — a code location
(`file:line`), a saved result JSON, or a deterministic **Reproduce:** command
(model weights are fixed and the scripts use no RNG, so commands reproduce exactly).
All measurement is local; the local driver reproduces production retrieval
byte-for-byte (verified — see §2).

---

## 0. Symptoms

- **lat-01 (exact citation):** "What was the final disposition in case 21-70493?"
  returned the named case only at **rank 5**, behind four off-topic policy cases.
  Artifact: `evaluation/latency/T_optimized_baseline_api_direct_*.json` (per-query
  `top_5_case_ids`), and the thematic sentinel row in
  `evaluation/thematic/baseline_results.json`.
- **Harm queries:** "sexual assault" returned policy-enumeration cases (`25-2581`,
  `19-16487`) and a **content-free signature block** (`16-73915`) above real
  sexual-assault claims. Artifact: live API top-5 (2026-06-09) and the baseline
  scorecard, `evaluation/thematic/baseline_results.json` row `A-sexual-assault`
  (core 14-70905 **ABSENT**).

---

## 1. What was ruled OUT (with the control that ruled it out)

### 1a. Prefix bug — RULED OUT (code read)
e5 requires asymmetric `"query: "` / `"passage: "` prefixes. The code applies them
correctly and symmetrically across ingest and serve:
- definition + application before tokenization: `rag_api/onnx_embedder.py:29-30`,
  `:56-57`.
- serve embeds the query via `embed_query` → `"query: "`:
  `rag_api/retrieval.py:323` (in `_embed_query`, reached by `search_dense:366`).
- ingest embeds chunks via `embed_passages` → `"passage: "`:
  `pipeline/migrate_to_qdrant.py:77-81` and `pipeline/rag_ingest.py:330`.
- **The cosine-1.0 parity test does NOT validate this** — it fed the *same* string
  to both embedders, so it only proves the ONNX forward matches sentence-transformers
  numerically, not that the right prefix is chosen per input. The code read is the
  independent confirmation.

### 1b. ONNX-export defect — RULED OUT (reference comparison)
The earlier "anisotropy might be an export bug" worry was argued from ONNX outputs
alone, which can't distinguish inherent vs export. The decisive test embeds the same
inputs through ONNX **and** stock sentence-transformers:
- ONNX ≡ sentence-transformers to **cosine 1.0000 on every input, including the
  empty string and the signature block** (not just contentful strings).
- Reproduce: `ninthc/bin/python` STEP-B script (loads `OnnxEmbedder` +
  `SentenceTransformer("intfloat/e5-small-v2")`; observed 2026-06-09):
  empty 1.0000, signature 1.0000, full chunk 1.0000.
- **Conclusion:** the degeneracy is in the e5 *weights*, not the conversion. No
  export work. The masking math is also correct for degenerate inputs
  (`onnx_embedder.py:81-84` — all prefix tokens unmasked).

### 1c. Embedder semantic ceiling — RULED OUT (vocabulary-mismatch probe)
Query "childhood molestation by a sibling" vs case `25-120` ("sexually abused by her
brother when she was a minor") shares **zero** content tokens → BM25 raw = **0.000**,
yet dense ranks `25-120` **#1**.
- Reproduce: STEP-3 P2 script; also the standing control `B-vocab-mismatch` in
  `evaluation/thematic/baseline_results.json` (core 25-120 @ rank 1).
- **Conclusion:** dense captures meaning with no lexical help — there is no general
  semantic ceiling.

### 1d. Chunk length — RULED OUT (filler control)
A generic 35-token non-legal sentence scored **lowest** (0.746) vs "sexual assault",
while the content-free signature (29 tok) scored **0.832** and an empty string
**0.836**.
- Reproduce: STEP-A script (observed 2026-06-09). The driver therefore filters on
  **information content, not length** (see §3).

---

## 2. The local driver is faithful to production (so the findings transfer)

`evaluation/thematic/score.py` → `LocalRetriever` mirrors production exactly:
dense-gated pool `max(20, k*5)` (`main.py:97` + `retrieval.py:428`), 0.6/0.4 blend
(`retrieval.py:467`), BM25 re-scored only within the dense pool (`retrieval.py:438-446`),
dedupe by `case_link` (`retrieval.py:471-479`; key matches `top_5_case_ids` at
`retrieval.py:198`). It reproduced the live k=5 top-5 for "sexual assault"
*exactly* (`25-2581, 19-16487, 16-73915, 25-120, 15-71553`) on 2026-06-09.
Parity was re-confirmed **post-deploy** against the live RRF serving path —
zero divergence on all six suite queries (§8).

---

## 3. The two actual diseases

### 3a. Boilerplate / empty-input hub-collapse (inherent e5 anisotropy)
Degenerate, low-information chunks collapse to a high-similarity "hub" vector and
out-score real content:
- empty string 0.836 and the `16-73915` signature 0.832 both **out-rank the
  genuinely-relevant full chunk 0.824** for "sexual assault" (STEP A).
- It is **information-vacancy, not length** (§1d). Reproduce: STEP-A script.
- This is the cause of `16-73915` ranking #3–#1 on harm queries.

### 3b. Attractor enumeration chunks (the dominant disease)
Long policy opinions (`25-2581` Immigrant Defenders, `19-16487` East Bay Sanctuary)
enumerate violence lists ("rape, kidnapping, sexual exploitation, assault") and score
high on **both** dense and BM25 for any harm query, so the 0.6/0.4 blend rewards them
twice and buries individual claims.
- Smoking gun: `14-70905` ("…raping her…") is **dense rank 5** yet **absent from the
  blend top-5** at both pool=25 and pool=200 — crowded out by the attractors.
  Reproduce: STEP-3 masking script (P1).
- Enlarging the pool does **not** fix it: gold coverage stays 1/4; pool 25→200 just
  swaps which single gold case appears (25-120 ↔ 17-73156). The pool bump is a
  band-aid, not a fix. Reproduce: STEP-3 P1 at pool 25 vs 200.

`hub_cos` cannot be used to filter 3a because it conflates boilerplate with
attractor-ness — the *real* East Bay Sanctuary chunk `18-17274` (cw=121) scores
hub=0.878, only 0.010 below the signature (0.888). Reproduce: STEP-a.2 distribution.

---

## 4. Hygiene (fix for 3a) — content-based, data-derived threshold

Threshold chosen from data, not guessed: **drop `content_words < 12`**.
- On the full corpus this isolates **exactly one chunk** — `16-73915 ch598`
  (cw=8) — with a clean +4 gap to the next, *real*, chunks (cw=12 `25-2581`,
  cw=13 `18-70505`'s ineffective-assistance holding). `hub_cos` rejected (gap 0.010,
  conflated). Reproduce: STEP-a.1/a.2 scripts.
- Gates passed: drop list = 1 chunk (eyeballed, pure boilerplate, zero false
  positives); per-case survival ✓ (`16-73915` 14→13 chunks, no case zeroed);
  P2 regression-guard ✓ (25-120 stays rank 1). Reproduce: STEP-a gates script.
- **Honest limit:** hygiene removes the signature pollution but buys **~0
  coverage lift on its own** — the attractors (disease 3b) are untouched.
  Reproduce: cleaned P1 still 1/2 gold at pool=25.

Hygiene is applied as the *substrate* for all thematic measurement (the local driver
drops cw<12). Deployed to production via a Qdrant re-ingest (DONE — collection
`asylum_cases_local_e5_384_onnx_clean`, 686 pts, verified; §8).

---

## 5. The thematic harness & frozen baseline

- Suite + graded gold (core/marginal/excluded, denominators) + pre-registered
  predictions: `evaluation/thematic/thematic_queries.json` (LOCKED, gold approved
  by the user). 6 queries across 3 classes; "religious persecution", FGM, and forced
  gang recruitment logged as **corpus coverage gaps** (0–1 real cases). Gold rulings
  traced per case in that file's `rationale`/`excluded` fields.
- Scorer: `evaluation/thematic/score.py` — per-query only (no blended number);
  `protect` rule, sentinel hard check, boundary-proximity flag.
- Frozen baseline (cleaned substrate, production ranking):
  `evaluation/thematic/baseline_results.json`.
  - A-harm **2/4** core (only political imprisonment 2/2; sexual assault and DV 0/0).
  - Two named leaks: **DV surfaces the perpetrator `21-70493` at #1** (a
    claim-agency/polarity failure, tracked as a named diagnostic, *not* expected to
    move with fusion); religious persecution (since dropped) surfaced enumeration noise.
  - Controls healthy (B 2/2 @ rank 1); **sentinel `21-70493` buried at rank 5**.

The earlier "1/4" figure was against an inflated 4-case gold; the honest
sexual-assault denominator is **2** (14-70905 core + 25-120 marginal), and 25-120 was
always recovered — the real gap is 14-70905.

---

## 6. Interventions tested

Adopted disqualification rule (user ruling): a protected core **falling out of
top-5** disqualifies; an intra-top-5 rank slip does not (top-5 is what the lawyer
reviews; order within it is below product resolution). A core at rank 5 is flagged
**boundary-fragile**.

### 6a. RRF + case-collapse, β=0 (pure max-per-case) — **WINNER**
Artifact: `evaluation/thematic/rrf_beta0_results.json`.
- **Sexual assault 0/1 → 1/1:** `14-70905` enters top-5 (rank 4, cleaned), attractors
  `25-2581`/`19-16487` **evicted**, `25-120` → #1.
- **Mechanism:** RRF is rank-based (`1/(60+rank)`), so it discards the attractors'
  both-signal *magnitude* dominance and keeps only rank position — the relevant case
  climbs in. This is a real fusion fix. Code: `retrieval.py:398-410` (`_fuse_rrf`).
- **Controls:** ✓ PASS (no protected core out); political imprisonment 2/2 (20-72806
  slips 3→4, allowed); both B-controls @ rank 1. **Sentinel held 5→5**
  (boundary-fragile, flagged).
- DV polarity **unfixed** (21-70493 still #1, 23-4420 absent) — as pre-registered;
  fusion does not do party-agency reasoning.

### 6b. RRF, β=0.1 (capped support) — **DROPPED**
Artifact: `evaluation/thematic/rrf_beta01_results.json`.
- Sexual assault regresses (14-70905 **absent** again; attractor 25-2581 returns #3),
  and breaks control `B-vocab-mismatch` (25-120 rank 1→3). **Mechanism:** support
  rewards multi-chunk cases = the long attractors, re-amplifying what RRF removed.

### 6c. gte-reranker (149M ModernBERT) stacked on β=0 RRF — **DO NOT DEPLOY**
Artifact: `evaluation/thematic/gte_rrf_beta0_results.json`. Pre-registered bar in
`thematic_queries.json` → `predictions."gte-reranker (stacked on β=0 RRF)"`.
- **PRIMARY (DV polarity) FAILED:** 23-4420 still absent, 21-70493 still #1, and gte
  pulled a *second* excluded case (14-70905) into DV top-5. The cross-encoder cannot
  distinguish DV *victim* from DV *perpetrator*.
- **Sentinel REGRESSION:** `21-70493` **dropped out of top-5** on the citation query —
  cross-encoders have no signal for a bare docket number (same lat-01 failure class).
- Secondary "win" (14-70905 4→3) is below product resolution and not clean (attractors
  return). Latency **p50 1133 ms / mean 2016 ms / max 6460 ms per query** over pool=25
  on local CPU — a 149M model that OOMs the 512 MB free tier and erases the latency
  win (server_total 1110→29 ms; `evaluation/latency/RESULTS.md`).

---

## 7. Recommendation

1. **RRF + case-collapse (β=0) — ✅ SHIPPED** (`FUSION_METHOD=rrf`, env knob
   `retrieval.py:303`; fusion impl `:398-410`). Live & verified — see §8.
2. **Content hygiene — ✅ SHIPPED** (drop cw<12; §4) via re-ingest to the clean
   Qdrant collection — firms 14-70905 to rank 4 and evicts the `16-73915` signature
   from #1. Verified live (§8).
3. **Do not deploy gte** or any cross-encoder (§6c): fails DV polarity, re-buries
   citations, can't fit free tier.
4. **Drop β>0 support** (§6b).

## 8. Deploy — SHIPPED & VERIFIED (2026-06-09, commit `c27fffd`)

**Deployed: the full validated stack** (RRF + case-collapse β=0 + content hygiene),
live on Render/Qdrant. Not gte, not β=0.1.

**What shipped:**
- Clean Qdrant collection `asylum_cases_local_e5_384_onnx_clean` — **686 points, dim
  384, Cosine; 30/30 case_links present, none zeroed** (re-ingest verified against
  live Qdrant). Prod `asylum_cases_local_e5_384_onnx` (687) **untouched as rollback**.
- Clean index dir committed (`c27fffd`): `data/experiments/T-optimized-onnx-clean/`
  (686-row parquet, `chunk_id` reset 0…685; force-added past the `data/*` gitignore;
  **regular file, not LFS** — 636,717 real bytes ship via `Dockerfile:16`).

**Matched env set (Render):** `VECTOR_STORE=qdrant` ·
`QDRANT_COLLECTION=asylum_cases_local_e5_384_onnx_clean` ·
`INDEX_DIR=/app/data/experiments/T-optimized-onnx-clean` · `FUSION_METHOD=rrf` ·
`USE_RERANKER=false`. **`INDEX_DIR` and `QDRANT_COLLECTION` are a matched pair** —
serving BM25 indexes `bm25_all[chunk_id]` (`retrieval.py:446`) off the `INDEX_DIR`
parquet, so both must point at the clean set together or BM25 misaligns.

**Live verification (`/health` + smoke, 2026-06-09):** `/health` =
`n_chunks:686, fusion_method:rrf, vector_store:qdrant, embedding_dim:384,
use_reranker:false, build_sha:c27fffd`. Serving top-5 reproduced the sim
(`evaluation/thematic/rrf_beta0_results.json`) **byte-for-byte**: "sexual assault" →
`[25-120, 15-71553, 24-2787, 14-70905, 20-72806]` (14-70905 @ **rank 4**, signature +
attractors gone); citation sentinel 21-70493 @ **rank 5**; controls 17-72197 #1 /
25-120 #1. `server_total` **15–60 ms** (no-NIM fast path intact). **sim↔serving
parity: zero divergence in production.**

**Rollback (one step, prod data intact):** set `FUSION_METHOD=blend` and repoint
`QDRANT_COLLECTION=asylum_cases_local_e5_384_onnx` +
`INDEX_DIR=/app/data/experiments/T-optimized-onnx` (the untouched 687-pt collection +
original parquet). No data restore needed.

**Config-only fallback (no re-ingest):** `FUSION_METHOD=rrf` alone on the dirty
687-chunk corpus also helps, but 14-70905 lands at fragile rank 5 and the `16-73915`
signature ranks #1 (reproduce: `score.py --fusion rrf --beta 0 --dirty`). The hygiene
collection shipped above is what firms 14-70905 to rank 4 and evicts the signature.

## 9. Unsolved / open

- **DV claim-agency/polarity:** no retrieval lever tested (blend, RRF, gte) surfaces
  the victim `23-4420` over the perpetrator `21-70493`. This needs a **claim-agency
  signal** (metadata: who is the asylum *applicant* vs. the convicted party), not
  fusion or reranking. Tracked as the named diagnostic in
  `thematic_queries.json` → `A-domestic-violence.diagnostic`.
- **Sentinel fragility:** the citation win sits at rank 5 (boundary-fragile). Exact
  docket lookups remain weak; a docket-number exact-match bypass is the targeted fix.
- **Corpus coverage gaps:** religious persecution, FGM, forced gang recruitment
  (`thematic_queries.json` → `coverage_gaps`).
