"""Per-request latency instrumentation for the RAG API.

A ContextVar holds one accumulator per in-flight request; a @timed decorator and
a timer() context manager record per-stage wall-clock into it. Nothing here
touches business logic — stages are instrumented at function/dispatch
boundaries, and every hook is a no-op when no request accumulator is active (so
direct unit-test calls and any non-handler caller are unaffected).

Network stages (embed/rerank/generate) are measured *gross* at their dispatch
point — which includes the retry backoff sleeps in nvidia_client._with_retry.
Those sleeps are recorded separately via record_sleep(); build_report() then
reports the sleep-excluded time as `<stage>_ms` and the sleep as
`<stage>_retry_sleep_ms`, so a 429-triggered 22s backoff is never miscounted as
network latency.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Iterator

# Network stages route through nvidia_client._with_retry, so their gross timing
# includes retry sleeps that build_report() subtracts back out.
_NETWORK_STAGES = ("embed", "rerank", "generate")
# Local CPU stages: gross == clean (no retry, no sleep).
_CPU_STAGES = ("dense_search", "bm25")


class Timings:
    """Mutable per-request accumulator. Times are milliseconds."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}    # gross wall-clock per timed stage
        self.sleeps: dict[str, float] = {}    # retry backoff per network call
        self.attempts: dict[str, int] = {}    # attempts made per network call
        self.context: dict[str, Any] = {}     # embedder_name, rerank_pool_size, …

    def add_stage(self, name: str, ms: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + ms

    def add_sleep(self, name: str, ms: float) -> None:
        self.sleeps[name] = self.sleeps.get(name, 0.0) + ms


_CURRENT: ContextVar[Timings | None] = ContextVar("rag_timings", default=None)


# ── Lifecycle ────────────────────────────────────────────────────────────────

def start() -> Timings:
    """Begin a fresh accumulator for this request and make it current."""
    t = Timings()
    _CURRENT.set(t)
    return t


def reset() -> None:
    """Clear the current accumulator (call in the handler's finally)."""
    _CURRENT.set(None)


def current() -> Timings | None:
    return _CURRENT.get()


# ── Recording hooks (all no-ops when no request is active) ───────────────────

@contextmanager
def timer(stage: str) -> Iterator[None]:
    """Record gross wall-clock of the wrapped block under `stage`."""
    t = _CURRENT.get()
    if t is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t.add_stage(stage, (time.perf_counter() - t0) * 1000.0)


def timed(stage: str) -> Callable:
    """Decorator form of timer() for whole-function stages."""
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t = _CURRENT.get()
            if t is None:
                return fn(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                t.add_stage(stage, (time.perf_counter() - t0) * 1000.0)
        return wrapper
    return deco


def record_sleep(call_name: str, ms: float) -> None:
    """Record retry backoff for a network call so it can be excluded from `<stage>_ms`."""
    t = _CURRENT.get()
    if t is not None:
        t.add_sleep(call_name, ms)


def record_attempts(call_name: str, attempts: int) -> None:
    """Record how many attempts a network call took (1 == no retry)."""
    t = _CURRENT.get()
    if t is not None:
        t.attempts[call_name] = attempts


def set_context(**fields: Any) -> None:
    """Attach non-timing context (embedder_name, rerank_pool_size, vector_store, …)."""
    t = _CURRENT.get()
    if t is not None:
        t.context.update(fields)


# ── Reporting ────────────────────────────────────────────────────────────────

def build_report(t: Timings, server_total_ms: float) -> dict[str, Any]:
    """Assemble the flat per-stage report.

    Network stages report sleep-excluded `<stage>_ms` plus `<stage>_retry_sleep_ms`
    and `<stage>_attempts`. CPU stages report their gross time. fusion_dedup_ms is
    the residual (server_total minus every *gross* stage), so it captures dict
    assembly + sort + dedup + guardrails + framework overhead without its own
    timer. By construction the reported components sum to ~server_total_ms.
    """
    report: dict[str, Any] = {}
    gross_sum = sum(t.stages.values())

    for stage in _NETWORK_STAGES:
        if stage not in t.stages:
            continue
        sleep = t.sleeps.get(stage, 0.0)
        report[f"{stage}_ms"] = round(max(0.0, t.stages[stage] - sleep), 2)
        report[f"{stage}_retry_sleep_ms"] = round(sleep, 2)
        report[f"{stage}_attempts"] = t.attempts.get(stage, 1)

    for stage in _CPU_STAGES:
        if stage in t.stages:
            report[f"{stage}_ms"] = round(t.stages[stage], 2)

    report["fusion_dedup_ms"] = round(max(0.0, server_total_ms - gross_sum), 2)
    report["server_total_ms"] = round(server_total_ms, 2)
    report.update(t.context)
    return report


def log_report(endpoint: str, report: dict[str, Any]) -> None:
    """Emit one JSON line per request to stdout for offline parsing."""
    print(json.dumps({"event": "request_timing", "endpoint": endpoint, **report},
                      sort_keys=True), flush=True)
