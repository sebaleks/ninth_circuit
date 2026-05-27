"""Guardrails: refuse low-relevance queries; sanity-check generated answers."""

from __future__ import annotations

REFUSAL_TEXT = "I can only answer about cases in our corpus."

# Refusal uses the DENSE cosine score (more uniformly calibrated than the
# rerank sigmoid, which collapses to 0 for natural-language questions).
# Empirical separation on the 30-case sample: in-corpus queries ≥0.21,
# out-of-corpus ≤0.10. Threshold tunable via evaluation.
MIN_DENSE_SCORE = 0.15


def should_refuse(dense_scores: list[float]) -> bool:
    """Return True if no retrieved chunk passes the dense-score threshold."""
    if not dense_scores:
        return True
    return max(dense_scores) < MIN_DENSE_SCORE


def is_refusal(answer: str) -> bool:
    """Did the model itself produce a refusal?"""
    return REFUSAL_TEXT.lower() in answer.lower()
