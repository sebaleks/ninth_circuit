"""Thin wrappers over NVIDIA NIM endpoints used by the RAG API.

All three live model IDs (May 2026, verified):
  embed:  nvidia/llama-nemotron-embed-1b-v2     (2048-dim, 2048-token context)
  rerank: nvidia/llama-nemotron-rerank-1b-v2
  gen:    meta/llama-3.3-70b-instruct

The user-specified llama-3.2-nv-{embedqa,rerankqa}-1b-v2 were end-of-lifed
2026-05-18; the Nemotron variants are the same-family successors.

Env vars:
  NVIDIA_API_KEY  — nvapi-... key
  NVIDIA_BASE_URL — defaults to https://integrate.api.nvidia.com/v1
                    (rerank endpoint lives under https://ai.api.nvidia.com/v1)
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import requests
from openai import APIStatusError, OpenAI


EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
EMBED_DIM = 2048
RERANK_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
GEN_MODEL = "meta/llama-3.3-70b-instruct"

DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
RERANK_BASE = "https://ai.api.nvidia.com/v1/retrieval"


def _api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    return key


def _client() -> OpenAI:
    return OpenAI(base_url=os.environ.get("NVIDIA_BASE_URL", DEFAULT_BASE), api_key=_api_key())


# ── Retry ────────────────────────────────────────────────────────────────────
# NIM's free tier returns 429 (rate limit) and transient 5xx (502/503/504) when
# its infrastructure is overloaded. Retry those a few times; fast-fail on
# anything else (auth, validation, payload). The three sleeps sum to 22s, which
# the eval's client-side timeout is sized to accommodate.

_RETRYABLE_STATUS = {429, 502, 503, 504}
_RETRY_SLEEPS = [2, 5, 15]  # backoff before attempts 2, 3, 4 → 4 attempts total


class _RetryableNimError(Exception):
    """Internal signal that a NIM call failed with a retryable HTTP status.

    Call sites raise this (carrying the originating exception) so the retry
    helper stays library-agnostic. On exhaustion the original exception is
    re-raised unchanged, so callers see the same type they do today.
    """

    def __init__(self, status_code: int, original: Exception):
        super().__init__(f"retryable NIM status {status_code}")
        self.status_code = status_code
        self.original = original


def _with_retry(call_name: str, fn):
    """Run `fn` with up to 4 attempts, backing off on `_RetryableNimError`.

    Any other exception propagates immediately (fast-fail). After the final
    attempt the original exception is re-raised.
    """
    for attempt in range(1, len(_RETRY_SLEEPS) + 2):  # 1, 2, 3, 4
        try:
            return fn()
        except _RetryableNimError as e:
            if attempt > len(_RETRY_SLEEPS):  # final attempt failed
                raise e.original from None
            sleep_s = _RETRY_SLEEPS[attempt - 1]
            print(
                f"nim retry: {call_name} got {e.status_code}, "
                f"sleeping {sleep_s}s (attempt {attempt}/4)…",
                file=sys.stderr,
            )
            time.sleep(sleep_s)


# ── Embedding ────────────────────────────────────────────────────────────────

def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed documents/passages. Returns (N, 2048) float32, L2-normalized."""
    return _embed(texts, input_type="passage")


def embed_query(text: str) -> np.ndarray:
    """Embed a single query. Returns (1, 2048) float32, L2-normalized."""
    return _embed([text], input_type="query")


def _embed(texts: list[str], input_type: str) -> np.ndarray:
    def _call():
        try:
            return _client().embeddings.create(
                model=EMBED_MODEL,
                input=texts,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "END"},
            )
        except APIStatusError as e:  # RateLimitError (429) subclasses this too
            if e.status_code in _RETRYABLE_STATUS:
                raise _RetryableNimError(e.status_code, e) from e
            raise

    resp = _with_retry("embed", _call)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


# ── Rerank ───────────────────────────────────────────────────────────────────

def rerank(query: str, passages: list[str]) -> list[float]:
    """Rerank passages against a query. Returns scores in [0,1] aligned to `passages`.

    Uses sigmoid over the model's raw logits so the score is absolute (not
    relative to the batch), which lets us use it for the refusal threshold.
    """
    if not passages:
        return []
    url = f"{RERANK_BASE}/{RERANK_MODEL}/reranking"
    payload = {
        "model": RERANK_MODEL,
        "query": {"text": query},
        "passages": [{"text": p} for p in passages],
    }
    def _call():
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in _RETRYABLE_STATUS:
                raise _RetryableNimError(status, e) from e
            raise
        return resp

    resp = _with_retry("rerank", _call)
    rankings = resp.json().get("rankings", [])

    # The API returns rankings sorted by relevance; map back to original order
    scores = [0.0] * len(passages)
    for r in rankings:
        idx = r["index"]
        logit = float(r["logit"])
        scores[idx] = 1.0 / (1.0 + math.exp(-logit))  # sigmoid
    return scores


# ── Generation ───────────────────────────────────────────────────────────────

GENERATION_SYSTEM = (
    "You are an assistant answering questions about Ninth Circuit asylum cases. "
    "Ground every claim in the numbered passages below. Cite the passages you used with "
    "bracketed tags like [1], [2] (one per claim is fine). "
    "If a passage is partially relevant, summarize what it does say and note the limit. "
    "Only respond with \"I can only answer about cases in our corpus.\" if the question "
    "is clearly unrelated to asylum, immigration, or the cited passages (e.g. weather, sports). "
    "Keep answers under 200 words."
)


def generate(question: str, passages: list[str], max_tokens: int = 500) -> str:
    """Single-shot generation. Returns the model's answer text."""
    numbered = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    user_msg = f"Passages:\n{numbered}\n\nQuestion: {question}\n\nAnswer (cite with [N]):"

    def _call():
        try:
            return _client().chat.completions.create(
                model=GEN_MODEL,
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
        except APIStatusError as e:  # RateLimitError (429) subclasses this too
            if e.status_code in _RETRYABLE_STATUS:
                raise _RetryableNimError(e.status_code, e) from e
            raise

    resp = _with_retry("generate", _call)
    return resp.choices[0].message.content.strip()
