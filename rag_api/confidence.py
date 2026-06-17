"""Two-signal confidence indicator (dense cosine + cross-encoder) → GREEN / YELLOW / RED.

A per-result demo indicator of how well the retrieved case TEXT matches the query — NOT a
guarantee of legal relevance. Robust by design: a 2-of-2 sum model, so any single-signal error
caps at YELLOW; RED needs BOTH signals weak.

Two signals, each scored 2/1/0 (High/Med/Low) via calibrated cutoffs, then summed:
    sum >= 3 → GREEN   (HH, HM, MH)
    sum == 2 → YELLOW  (MM, HL, LH)
    sum <= 1 → RED     (ML, LM, LL)

Calibration (jina-reranker-v1-tiny-en CE; abstention probe set — see
evaluation/nim_e2e/results/ABSTENTION_CROSSENCODER_FINDINGS.md):
  DENSE cosine — off-topic ≤0.147, in-corpus p05=0.264 / p25=0.325 / median=0.366, bulk 0.33–0.56.
  CE (jina-tiny) — in-corpus median 2.07 (p05=1.58), out_vocab median 1.25.
CONSERVATIVE LOW-END: CE M/L = 1.30 < the DV anchor (a real DV query the CE under-scored at 1.47),
so that real-but-CE-underscored passage lands Med, not Low → with its High dense it is GREEN, and
in no case can a single CE error alone push a real result to RED. Under-warn over over-warn.

CE fires only on the dense BAND [0.15, 0.50], non-lookup traffic (the uncertain middle); dense<0.15
is the abstention floor (refused, no color), dense>0.50 is confident (dense alone), and docket/cite
"lookup" queries route AROUND the CE (it collapses on them) and are colored by retrieval strength.
"""
from __future__ import annotations
import re

# ── calibrated cutoffs (model-specific to jina-reranker-v1-tiny-en) ──
DENSE_HM, DENSE_ML = 0.33, 0.15      # dense: High≥0.33, Med[0.15,0.33), Low<0.15
CE_HM, CE_ML = 1.80, 1.30            # CE:    High≥1.80, Med[1.30,1.80), Low<1.30
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
