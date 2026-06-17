"""Two-signal confidence indicator (dense cosine + cross-encoder) → GREEN / YELLOW / RED.

A per-result demo indicator of how well the retrieved case TEXT matches the query — NOT a
guarantee of legal relevance. Robust by design: a 2-of-2 sum model, so any single-signal error
caps at YELLOW; RED needs BOTH signals weak.

Two signals, each scored 2/1/0 (High/Med/Low) via calibrated cutoffs, then summed:
    sum >= 3 → GREEN   (HH, HM, MH)
    sum == 2 → YELLOW  (MM, HL, LH)
    sum <= 1 → RED     (ML, LM, LL)

Calibration (ms-marco-MiniLM-L-6-v2 CE @ top-5; abstention probe set — see
evaluation/nim_e2e/results/CONFIDENCE_MEMORY_FINDINGS.md for the L-6 memory/model decision and
ABSTENTION_CROSSENCODER_FINDINGS.md for the eval):
  DENSE cosine — off-topic ≤0.147, in-corpus p05=0.264 / p25=0.325 / median=0.366, bulk 0.33–0.56.
  CE (L-6) — in-corpus in-band median ~5.7 (p25=3.70), out_vocab in-band p50=1.05 / p90=3.47.
CONSERVATIVE LOW-END: CE M/L = 0.80 < the DV anchor (a real DV query the L-6 CE under-scored at
1.00), so that real-but-CE-underscored passage lands Med, not Low → with its High dense it is GREEN,
and in no case can a single CE error alone push a real result to RED. Under-warn over over-warn.
(L-6 chosen over the eval's jina-tiny on memory: jina-tiny OOMs the 512 MB box; L-6 @ top-5 fits at
475 MB and catches 62.5% of out_vocab. Cutoffs are L-6-specific — they do NOT transfer across CE models.)

CE fires only on the dense BAND [0.15, 0.50], non-lookup traffic (the uncertain middle); dense<0.15
is the abstention floor (refused, no color), dense>0.50 is confident (dense alone), and docket/cite
"lookup" queries route AROUND the CE (it collapses on them) and are colored by retrieval strength.
"""
from __future__ import annotations
import re

# ── calibrated cutoffs (model-specific to ms-marco-MiniLM-L-6-v2 @ top-5) ──
DENSE_HM, DENSE_ML = 0.33, 0.15      # dense: High≥0.33, Med[0.15,0.33), Low<0.15
CE_HM, CE_ML = 3.70, 0.80            # L-6 CE: High≥3.70, Med[0.80,3.70), Low<0.80
BAND_LO, BAND_HI = 0.15, 0.50        # CE adjudicates only this dense band

GREEN, YELLOW, RED = "green", "yellow", "red"
LABEL = {GREEN: "Strong match", YELLOW: "Moderate match", RED: "Weak match"}
TOOLTIP = ("Confidence reflects how well the case text matches your query — not a guarantee of "
           "legal relevance. Review the case.")

_DOCKET = re.compile(r"\b\d{2}-\d{2,5}\b")

def is_lookup(query: str) -> bool:
    """Docket / case-name / statute query → route around the CE (BM25 serves these; the CE
    under-scores them). Token-based router (matched the idealized category-router in eval)."""
    ql = query.lower()
    return (bool(_DOCKET.search(query)) or " v. " in ql or "matter of" in ql
            or "u.s.c" in ql or "§" in query or "c.f.r" in ql)

def _tier(score: float, hm: float, ml: float) -> int:
    return 2 if score >= hm else (1 if score >= ml else 0)

def _color_from_sum(s: int) -> str:
    return GREEN if s >= 3 else (YELLOW if s == 2 else RED)

def confidence(dense_score: float, ce_score: float | None, query: str,
               exact_match: bool = False) -> dict:
    """Per-result indicator. Returns {color, label, signals, dense_tier, ce_tier?, sum?}.

    - LOOKUP query: colored by retrieval strength. An EXACT match (the queried docket/authority is
      in the top result) → GREEN regardless of dense; otherwise the dense tier is the proxy. CE not
      used (it under-scores lookups, so it is routed around).
    - dense outside [0.15,0.50]: CE not fired → dense alone (High→GREEN, Med→YELLOW, Low→RED).
    - in band, non-lookup: full 2-of-2 (dense tier + CE tier), summed.
    """
    d = _tier(dense_score, DENSE_HM, DENSE_ML)
    lookup = is_lookup(query)
    if lookup and exact_match:
        return {"color": GREEN, "label": LABEL[GREEN], "signals": "lookup/exact", "dense_tier": d}
    use_ce = (ce_score is not None) and (not lookup) and (BAND_LO <= dense_score <= BAND_HI)
    if not use_ce:
        color = {2: GREEN, 1: YELLOW, 0: RED}[d]
        return {"color": color, "label": LABEL[color],
                "signals": "lookup/dense-only" if lookup else "dense-only", "dense_tier": d}
    c = _tier(ce_score, CE_HM, CE_ML)
    color = _color_from_sum(d + c)
    return {"color": color, "label": LABEL[color], "signals": "dense+ce",
            "dense_tier": d, "ce_tier": c, "sum": d + c}
