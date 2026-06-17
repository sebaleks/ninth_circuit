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
