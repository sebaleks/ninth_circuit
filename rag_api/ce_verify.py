"""Opt-in, streaming "Result Cross Verification" — the L-6 cross-encoder kept OUT of the default path.

The CE is CPU-bound on the 512 MB Render free tier (~1 s per (query, passage) pair, ~10 s for 10),
so this module:
  - loads the model ONLY on explicit Enable (lazy, idempotent, locked),
  - scores ONLY on explicit Cross Verify, one result at a time,
  - is halted by client disconnect (the SSE loop checks between results) so a Stop actually frees CPU,
  - unloads on Disable / page-close to release the memory.

The default /search path never touches this module, so baseline latency (~255 ms) and footprint stay
low. Thresholds are L-6-specific (ms-marco-MiniLM-L-6-v2) — do NOT transfer them to another CE.
"""
from __future__ import annotations

import gc
import os
import threading

from rag_api import confidence as conf

CE_MODEL = os.environ.get("CE_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
MAX_PASSAGE_CHARS = 1800   # snippet truncation fed to the CE
BATCH = 5                  # batch size (peak memory == one 5-pair batch; arena disabled)
MAX_SCORE = 10             # HARD CAP: never CE-score more than this many results per verify
TARGET_NONGREY = 5         # backfill target: maintain ~this many non-greyed results

# Calibrated tiers (reuse the L-6 constants from confidence.py): DENSE high>=0.33, med[0.15,0.33);
# CE high>3.70, med(0.80,3.70], low(0,0.80]; plus a CE<=0 HARD OVERRIDE -> "Not relevant"/grey.
_DENSE_HM, _DENSE_ML = conf.DENSE_HM, conf.DENSE_ML  # 0.33, 0.15
_CE_HM, _CE_ML = conf.CE_HM, conf.CE_ML              # 3.70, 0.80

_model = None
_lock = threading.Lock()


def is_loaded() -> bool:
    return _model is not None


def load() -> None:
    """Lazy-load the CE (Enable). Idempotent + locked so concurrent enables load exactly one copy."""
    global _model
    with _lock:
        if _model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazy: only when enabled
            _model = TextCrossEncoder(
                model_name=CE_MODEL, threads=1,
                extra_session_options={"enable_cpu_mem_arena": False},
            )


def unload() -> None:
    """Free the CE (Disable / page-close) and reclaim its memory."""
    global _model
    with _lock:
        _model = None
    gc.collect()


def score(query: str, snippet: str) -> float:
    """One (query, passage) raw CE logit. Requires a loaded model (raises otherwise)."""
    m = _model
    if m is None:
        raise RuntimeError("CE not loaded — call load() first")
    return float(next(iter(m.rerank(query, [snippet[:MAX_PASSAGE_CHARS]]))))


def label_for(dense_score: float, ce_score: float) -> dict:
    """Calibrated label, checked in order:

    1. CE raw score <= 0  -> grey "Not relevant" (KEEP VISIBLE) — a hard override: a non-positive
       logit means the CE says "no" even if dense rated it relevant (vetoing dense's false positives).
    2. else color = dense_tier + ce_tier (2-of-2 sum): >=3 GREEN "Strong" / 2 YELLOW "Moderate" /
       <=1 RED "Weak".
    """
    if ce_score <= 0.0:
        return {"color": "grey", "label": "Not relevant", "treatment": "grey"}
    d = 2 if dense_score >= _DENSE_HM else (1 if dense_score >= _DENSE_ML else 0)
    c = 2 if ce_score > _CE_HM else (1 if ce_score > _CE_ML else 0)
    s = d + c
    if s >= 3:
        return {"color": "green", "label": "Strong", "treatment": "normal"}
    if s == 2:
        return {"color": "yellow", "label": "Moderate", "treatment": "normal"}
    return {"color": "red", "label": "Weak", "treatment": "normal"}


def applies(query: str, hits: list[dict]) -> bool:
    """Cross Verify is meaningful only for band, non-lookup queries: top dense in [0.15, 0.50] and
    not a docket/cite lookup. Confident (top dense > 0.50) or lookup queries -> not applicable."""
    if conf.is_lookup(query) or not hits:
        return False
    top = max(float(h.get("dense_score", h.get("score", 0.0))) for h in hits)
    return conf.BAND_LO <= top <= conf.BAND_HI
