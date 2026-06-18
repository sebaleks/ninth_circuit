# Result Cross Verification — opt-in, streaming CE (design + validation)

Branch `feat/cross-verify-stream`. Reworks the cross-encoder from always-on (the old
CONFIDENCE_ENABLED inline path, which made band queries ~10 s on the free tier — the CE is CPU-bound
at ~1 s/pair) into an **opt-in, load-on-demand, SSE-streamed** feature, so the default path stays fast.

## Three states
- **Default:** CE NOT loaded. `/search` and `/chat` run dense + Qdrant-BM25 + union-RRF only →
  **~255 ms**, no labels. Baseline footprint low (no CE in memory).
- **Enable** (`POST /verify/enable`): lazy-load the L-6 CE (one-time memory + load cost; locked +
  idempotent so concurrent enables load one copy). Unload on **Disable** (`POST /verify/disable`)
  or page-close (`navigator.sendBeacon`).
- **Cross Verify** (`GET /verify/stream?query=&k=5`, SSE): score the current results one-by-one in
  rank order, streaming a label per result.

## SSE contract
`data:` JSON per line. `result` event: `{rank, result_id(=case_link), label, color, treatment,
ce_score, dense_score, case_link, snippet, page, case_disposition, case_pub_status}`. Also
`not_applicable` (lookup / high-confidence), `error{reason:not_loaded}`, `done{scored, nongrey}`.

## Labels (L-6 calibrated — do NOT transfer to another CE), checked in order
1. **CE raw ≤ 0 → grey "Not relevant"** (kept visible, de-emphasized) — hard override: a non-positive
   logit is the CE vetoing dense's false positive.
2. else color = dense_tier + ce_tier (2-of-2): ≥3 GREEN "Strong" / 2 YELLOW "Moderate" / ≤1 RED "Weak".
   - dense: high ≥0.33, med [0.15,0.33). CE: high >3.7, med (0.8,3.7], low (0,0.8].

## Backfill + hard cap
After the top-5, if any are greyed, score ranks 6,7,… to maintain **5 non-grey**, hard-capped at
**10 scored total** (CPU is the constraint). Backfilled results stream as new cards.

## Halt (frees CPU)
The SSE loop checks `request.is_disconnected()` **between each result** and breaks. Closing the
EventSource (Stop / navigate away) disconnects the request → the loop stops → no more CE calls.
At most one in-flight result completes after the click (per-result granularity).

## Validation (probe set, L-6) — PASS
Applied the label scheme to the verify-eligible (band, non-lookup) probes:
- **Real queries (in_thematic + in_edge): 20/20 Strong/Moderate, 0/20 greyed.** The CE≤0 override
  never fires on a real query — grey is safe (the load-bearing conservative case).
- **out_vocab: 10/30 greyed "Not relevant"** (the clearly-negative ones). The other 20/30 (vampire/
  Wakanda-style elaborate adversaries) still score CE>0 → Strong/Moderate — L-6's known ~62.5%
  ceiling, unchanged. Those would NOT be greyed (shown with a label), consistent with "leave the
  clever cases to the indicator, not a hard hide".
- All exact label anchors reproduce (real-strong→Strong, DV-underscored→Strong, out_vocab CE<0→grey).
- 19 backend tests (label calibration, applies, lifecycle, stream backfill/cap/not-loaded/
  not-applicable); 126 total pass.

## Known limitation
Concurrent "Enable" shares ONE global model (load-once); but one user's Disable unloads it for all
(low-traffic MVP — acceptable; the stream returns `error:not_loaded` and the UI re-prompts Enable).
Per-session copies or refcounting are optional hardening, not built.

## Gate
Backend done + probe-validated. Frontend (Enable/Disable/Cross-Verify/Stop + SSE proxy with abort
propagation + grey rendering) in progress. **Not deployed; production (main, sha 6b46547) unchanged.**
Before merge: confirm the live SSE halt actually frees CPU (close stream mid-score → backend stops).
