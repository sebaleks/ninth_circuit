"""Cross-encoder confidence annotator (L-6 @ top-5). Lazy, memory-bounded, env-gated.

Attaches `dense_score`, `ce_score`, and a `confidence` {color,label,tooltip} to each search hit.
The CE (ms-marco-MiniLM-L-6-v2 via fastembed/onnxruntime) jointly encodes (query, passage) to
flag corpus-vocabulary nonsense the bi-encoder cosine cannot (see CONFIDENCE_MEMORY_FINDINGS.md).

It fires ONLY on the dense band [0.15,0.50] of NON-lookup queries — the uncertain middle; docket/
cite lookups route around it (the CE under-scores them) and confident queries (dense>0.50) don't
need it. Scoring is batched at BATCH passages with onnxruntime's arena disabled and threads=1, so
peak RSS stays ~the validated 475 MB regardless of k. Disabled unless CONFIDENCE_ENABLED=true, so
the default deployment is byte-for-byte unchanged.
"""
from __future__ import annotations

import os
import re

from rag_api import confidence as conf

CE_MODEL = os.environ.get("CE_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
BATCH = 5                  # memory bound: one CE batch == the validated top-5 footprint
MAX_PASSAGE_CHARS = 1800   # ~300 words; caps sequence length (memory + the eval setting)
_CE = None
_DOCKET = re.compile(r"\b\d{2}-\d{2,5}\b")


def enabled() -> bool:
    """Feature flag. Default off → no CE load, no behavior change for existing deployments."""
    return os.environ.get("CONFIDENCE_ENABLED", "false").strip().lower() == "true"


def refuse_enabled() -> bool:
    """CE-based abstention flag — separate from coloring so refusal can be toggled independently
    (and instantly killed). Requires CONFIDENCE_ENABLED too, since it reads the CE scores."""
    return os.environ.get("CE_REFUSE_ENABLED", "false").strip().lower() == "true"


def should_refuse(query: str, hits: list[dict]) -> bool:
    """CE abstention for the dense-uncertain band — the gap the 0.15 dense floor can't close.

    Refuse a band [0.15, 0.50], non-lookup query when even its BEST result is an active non-match
    (max CE < CE_REFUSE_FLOOR, default 0.0) — i.e. the cross-encoder scores every returned passage
    as "does not answer this query". This catches corpus-vocabulary nonsense whose top dense cosine
    happens to clear 0.15 ("george washington denied slavery" → max CE ≈ −10) without the dense-only
    false-refuses a higher cosine threshold would cause. Floor 0 is conservative: real in-corpus
    queries keep ≥ ~1.0 of margin (their best passage scores well above 0), so this should not
    refuse real queries — while leaving the genuinely-hard adversaries (which score positive) to the
    RED indicator rather than a hard refuse.

    No-op unless CE_REFUSE_ENABLED. Never fires for: lookups (routed around the CE), confident
    queries (top dense > 0.50), dense-already-refused queries (< 0.15, the dense gate owns those),
    or when the CE didn't run. Caller must have run annotate() first so ce_score is populated.
    """
    if not refuse_enabled() or not hits or conf.is_lookup(query):
        return False
    dense = [float(h.get("dense_score", h.get("score", 0.0))) for h in hits]
    top = max(dense)
    if top < conf.BAND_LO or top > conf.BAND_HI:  # <0.15 → dense gate; >0.50 → confident, serve
        return False
    ce = [h["ce_score"] for h in hits if h.get("ce_score") is not None]
    if not ce:  # CE didn't fire — don't refuse on a missing signal
        return False
    floor = float(os.environ.get("CE_REFUSE_FLOOR", "0.0"))
    return max(ce) < floor


def _ce():
    """Lazy CE singleton — built on first annotate(), never at import (memory)."""
    global _CE
    if _CE is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazy: only when enabled
        _CE = TextCrossEncoder(
            model_name=CE_MODEL, threads=1,
            extra_session_options={"enable_cpu_mem_arena": False},
        )
    return _CE


def _exact_lookup_match(query: str, hit: dict) -> bool:
    """A docket token in the query that appears in this hit's case_link → exact lookup match."""
    link = (hit.get("case_link") or "").lower()
    return any(d in link for d in _DOCKET.findall(query))


def _score_batched(query: str, passages: list[str]) -> list[float]:
    """CE-score passages in BATCH-sized groups so peak memory == one batch, not all of k."""
    ce = _ce()
    out: list[float] = []
    for i in range(0, len(passages), BATCH):
        out.extend(float(s) for s in ce.rerank(query, passages[i:i + BATCH]))
    return out


def annotate(query: str, hits: list[dict]) -> list[dict]:
    """Attach dense_score / ce_score / confidence to each hit (in place) and return it.

    The CE fires once (batched) only for non-lookup queries that have at least one in-band
    result; otherwise ce_score stays None and confidence falls back to dense-only / lookup
    coloring. Caller should guard with enabled().
    """
    lookup = conf.is_lookup(query)
    dense = [float(h.get("dense_score", h.get("score", 0.0))) for h in hits]
    ce_scores: list[float | None] = [None] * len(hits)
    if hits and not lookup and any(conf.BAND_LO <= d <= conf.BAND_HI for d in dense):
        passages = [(h.get("snippet") or "")[:MAX_PASSAGE_CHARS] for h in hits]
        ce_scores = list(_score_batched(query, passages))
    for h, d, ce in zip(hits, dense, ce_scores):
        em = _exact_lookup_match(query, h) if lookup else False
        c = conf.confidence(d, ce, query, exact_match=em)
        h["dense_score"] = d
        h["ce_score"] = ce
        h["confidence"] = {"color": c["color"], "label": c["label"], "tooltip": conf.TOOLTIP}
    return hits
