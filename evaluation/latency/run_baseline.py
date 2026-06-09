#!/usr/bin/env python3
"""Baseline latency measurement for the RAG /search endpoint.

Runs one warmup query (discarded), then a fixed 10-query set sequentially
against a live deployment with a pause between each (to avoid NIM rate limits).
For every query it captures the server-side per-stage timings (via
?include_timings=true) plus the client-side wall-clock, then reports p50/p95/mean
per stage over the successful queries.

Talks to the deployed HTTP API only — it does NOT import the rag_api package, so
it can be pointed at prod, a preview, or a local server interchangeably.

NOTE: the per-stage server timings require the latency-instrumentation build
(the `timings` field + ?include_timings) to be deployed at --rag-url. Against a
deployment without it, the script still records e2e_ms and marks server stages
as null (and prints a warning).

Usage:
    python evaluation/latency/run_baseline.py
    python evaluation/latency/run_baseline.py --rag-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

HERE = Path(__file__).resolve().parent
# Full endpoint URL to POST against (NOT a base URL). Both conventions are just
# "full URL": Render-direct ".../search", Vercel proxy ".../api/rag/search".
DEFAULT_RAG_URL = "https://ninth-circuit.onrender.com/search"
DEFAULT_LABEL = "custom"  # used when the URL host matches no known pattern
DEFAULT_QUERIES = HERE / "baseline_queries.json"
DEFAULT_PAUSE_SECONDS = 4.0
# Render free-tier cold start can take tens of seconds; the two NIM round-trips
# (embed + rerank) add ~1.4s on top. Generous so a slow first hit isn't an error.
REQUEST_TIMEOUT_S = 60
K = 5
WARMUP_QUERY_ID = "lat-01"

# Stage fields aggregated into p50/p95/mean. The first group comes from the
# server `timings` payload; the last two are client-side (see measure_query).
STAGE_FIELDS = [
    "embed_ms",
    "dense_search_ms",
    "rerank_ms",
    "bm25_ms",
    "fusion_dedup_ms",
    "server_total_ms",
    "e2e_ms",
    "network_overhead_ms",
]
CONTEXT_FIELDS = ["embedder_name", "embedding_dim", "rerank_pool_size", "vector_store"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def with_include_timings(url: str) -> str:
    """Return `url` with include_timings=true added to its query string.

    Adds `?include_timings=true` when there's no query, `&include_timings=true`
    when there already is — handled cleanly by parsing and re-encoding rather
    than string-concatenation.
    """
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query["include_timings"] = "true"
    return urlunsplit(parts._replace(query=urlencode(query)))


def label_for(url: str, fallback: str) -> str:
    """Derive the run label from the target URL host."""
    if "onrender.com" in url:
        return "api_direct"
    if "vercel.app" in url:
        return "vercel_proxy"
    return fallback


def git_sha() -> str | None:
    """Current HEAD sha, or None if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001 — best-effort metadata only
        return None


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile (numpy default method). None if empty."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def load_queries(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Queries file not found: {path}")
    queries = json.loads(path.read_text())
    if not isinstance(queries, list) or not queries:
        sys.exit(f"Queries file must be a non-empty JSON list: {path}")
    return queries


def stage_value(record: dict, field: str):
    """Pull a stage value: client-side fields off the record, the rest from timings."""
    if field in ("e2e_ms", "network_overhead_ms"):
        return record.get(field)
    return (record.get("server_timings") or {}).get(field)


# ── Measurement ──────────────────────────────────────────────────────────────

def measure_query(rag_url: str, q: dict, timeout: float) -> dict:
    """POST one query to /search?include_timings=true and build its record.

    Never raises: connection/timeout/HTTP/parse failures are captured as a record
    with status="error" so the run continues.
    """
    # rag_url is the full endpoint; append include_timings to whatever it is.
    url = with_include_timings(rag_url)
    record: dict = {"id": q.get("id"), "query": q.get("query")}

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            url,
            json={"query": q["query"], "k": K},
            timeout=timeout,
        )
    except requests.RequestException as e:
        record.update(
            status="error",
            error_msg=f"{type(e).__name__}: {e}",
            e2e_ms=round((time.perf_counter() - t0) * 1000, 2),
            server_timings=None,
            context=None,
            network_overhead_ms=None,
            top_5_case_ids=[],
        )
        return record

    e2e_ms = (time.perf_counter() - t0) * 1000

    if resp.status_code != 200:
        record.update(
            status="error",
            error_msg=f"HTTP {resp.status_code}: {resp.text[:200]}",
            e2e_ms=round(e2e_ms, 2),
            server_timings=None,
            context=None,
            network_overhead_ms=None,
            top_5_case_ids=[],
        )
        return record

    try:
        body = resp.json()
    except ValueError as e:
        record.update(
            status="error",
            error_msg=f"invalid JSON: {e}",
            e2e_ms=round(e2e_ms, 2),
            server_timings=None,
            context=None,
            network_overhead_ms=None,
            top_5_case_ids=[],
        )
        return record

    timings = body.get("timings") or {}
    server_total = timings.get("server_total_ms")
    record.update(
        status="refused" if body.get("refused") else "ok",
        server_timings=timings,
        context={k: timings.get(k) for k in CONTEXT_FIELDS},
        e2e_ms=round(e2e_ms, 2),
        # Positive ⇒ network + Render queueing time outside the server. None when
        # the target didn't return server timings (no instrumentation deployed).
        network_overhead_ms=(round(e2e_ms - server_total, 2)
                             if isinstance(server_total, (int, float)) else None),
        top_5_case_ids=[h.get("case_link") for h in body.get("hits", [])][:K],
    )
    return record


def aggregate(records: list[dict]) -> dict:
    """p50/p95/mean per stage, computed over status=='ok' records only."""
    successful = [r for r in records if r.get("status") == "ok"]
    agg: dict = {}
    for field in STAGE_FIELDS:
        vals = [v for r in successful if (v := stage_value(r, field)) is not None]
        if vals:
            agg[field] = {
                "p50": round(percentile(vals, 50), 2),
                "p95": round(percentile(vals, 95), 2),
                "mean": round(sum(vals) / len(vals), 2),
                "n": len(vals),
            }
        else:
            agg[field] = {"p50": None, "p95": None, "mean": None, "n": 0}
    return agg


# ── Output ───────────────────────────────────────────────────────────────────

def print_summary(agg: dict, meta: dict) -> None:
    print()
    print(f"Baseline latency vs {meta['rag_url']}")
    print(f"  {meta['n_successful']}/{meta['n_queries']} ok, "
          f"{meta['n_refused']} refused, {meta['n_errors']} error(s)")
    if not meta.get("server_timings_available", False):
        print("  WARNING: target returned no server `timings` — per-stage rows are "
              "null. Deploy the latency-instrumentation build, or point --rag-url "
              "at a local server.")
    print()
    header = f"{'stage':<22}{'p50':>10}{'p95':>10}{'mean':>10}"
    print(header)
    print("-" * len(header))
    for field in STAGE_FIELDS:
        a = agg.get(field, {})
        cells = "".join(
            f"{a.get(key):>10.2f}" if isinstance(a.get(key), (int, float)) else f"{'-':>10}"
            for key in ("p50", "p95", "mean")
        )
        print(f"{field:<22}{cells}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rag-url", default=DEFAULT_RAG_URL,
                        help="FULL endpoint URL to POST against (not a base URL), e.g. "
                             "https://host/search or https://host/api/rag/search "
                             f"(default: {DEFAULT_RAG_URL})")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES,
                        help=f"path to queries JSON (default: {DEFAULT_QUERIES})")
    parser.add_argument("--output", type=Path, default=None,
                        help="path to write results (default: T1_baseline_<label>_<timestamp>.json)")
    parser.add_argument("--pause-seconds", type=float, default=DEFAULT_PAUSE_SECONDS,
                        help=f"pause between queries (default: {DEFAULT_PAUSE_SECONDS})")
    parser.add_argument("--label", default=DEFAULT_LABEL,
                        help="output-filename label, used only when the URL host matches no "
                             "known pattern (onrender.com -> api_direct, vercel.app -> "
                             f"vercel_proxy; default: {DEFAULT_LABEL})")
    args = parser.parse_args()

    label = label_for(args.rag_url, args.label)
    output_path = args.output or (HERE / f"T1_baseline_{label}_{ts}.json")
    queries = load_queries(args.queries)
    warmup_q = next((q for q in queries if q.get("id") == WARMUP_QUERY_ID), queries[0])

    print(f"Target:  {args.rag_url}  [label: {label}]")
    print(f"Queries: {len(queries)} from {args.queries} (warmup: {warmup_q.get('id')})")
    print(f"Pause:   {args.pause_seconds}s between queries")
    print()

    # ── Warmup (discarded) ──
    print(f"[warmup] {warmup_q.get('id')} …", flush=True)
    warm = measure_query(args.rag_url, warmup_q, REQUEST_TIMEOUT_S)
    print(f"[warmup] {warm['status']} in {warm.get('e2e_ms')}ms (discarded)")
    time.sleep(args.pause_seconds)

    # ── Measured run ──
    records: list[dict] = []
    for i, q in enumerate(queries):
        rec = measure_query(args.rag_url, q, REQUEST_TIMEOUT_S)
        records.append(rec)
        server_total = (rec.get("server_timings") or {}).get("server_total_ms")
        print(f"[{i + 1:>2}/{len(queries)}] {rec.get('id')} {rec['status']:<7} "
              f"e2e={rec.get('e2e_ms')}ms server_total={server_total}"
              + (f"  ({rec.get('error_msg')})" if rec.get("error_msg") else ""),
              flush=True)
        if i < len(queries) - 1:
            time.sleep(args.pause_seconds)

    n_successful = sum(1 for r in records if r["status"] == "ok")
    n_refused = sum(1 for r in records if r["status"] == "refused")
    n_errors = sum(1 for r in records if r["status"] == "error")
    timings_available = any(r.get("server_timings") for r in records if r["status"] == "ok")

    metadata = {
        "rag_url": args.rag_url,
        "label": label,
        "output_path": str(output_path),
        "queries_path": str(args.queries),
        "n_queries": len(queries),
        "warmup_query": warmup_q.get("id"),
        "pause_seconds": args.pause_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "n_successful": n_successful,
        "n_refused": n_refused,
        "n_errors": n_errors,
        "server_timings_available": timings_available,
    }
    aggregates = aggregate(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        {"metadata": metadata, "aggregates": aggregates, "queries": records},
        indent=2, sort_keys=False,
    ))

    print_summary(aggregates, metadata)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
