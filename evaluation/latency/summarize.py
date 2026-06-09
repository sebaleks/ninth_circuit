#!/usr/bin/env python3
"""Summarize latency result JSONs into a single comparison table.

Scans evaluation/latency/ for result files (baseline_*.json / test_*.json),
pulls each run's configuration + headline stage timings, and renders one
markdown table to stdout (and optionally to RESULTS.md). A Δ column shows each
run's server_total p50 relative to the chosen baseline (T1).

Reads the JSON shape written by run_baseline.py:
    {"metadata": {...}, "aggregates": {stage: {p50,p95,mean,n}}, "queries": [...]}
Per-run configuration (embedder_name, embedding_dim, vector_store) lives in each
query's `context`; reranker_name / fusion_method are not emitted by the current
harness and therefore render as "—" until a future run records them in metadata.

Missing fields degrade to "—" rather than crashing, so older/partial runs still
list. Files that aren't result JSONs (e.g. baseline_queries.json) are skipped.

Usage:
    python evaluation/latency/summarize.py
    python evaluation/latency/summarize.py --md-file
    python evaluation/latency/summarize.py --baseline baseline_run2
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_MD = HERE / "RESULTS.md"

# Result-file globs. Edit if the naming convention changes.
PATTERNS = ["baseline_*.json", "test_*.json"]

# Optional explicit test ordering by test_id; ids not listed fall to the end,
# alphabetically. Leave empty for purely alphabetical ordering.
TEST_ORDER: list[str] = []

# Trailing run timestamp (…_20260609T182605Z) stripped to form a clean test_id.
_TS_SUFFIX = re.compile(r"_\d{8}T\d{6}Z$")

# Config fields surfaced as columns (label, json key).
CONFIG_COLUMNS = [
    ("Embedder", "embedder_name"),
    ("Dim", "embedding_dim"),
    ("Store", "vector_store"),
    ("Reranker", "reranker_name"),
    ("Fusion", "fusion_method"),
]


# ── Loading ──────────────────────────────────────────────────────────────────

def discover_files(directory: Path) -> list[Path]:
    seen: dict[Path, None] = {}
    for pattern in PATTERNS:
        for p in sorted(directory.glob(pattern)):
            seen.setdefault(p, None)
    return list(seen)


def load_result(path: Path) -> dict | None:
    """Parse a result file into a normalized row dict, or None if it isn't one."""
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    # A result file is a dict with an "aggregates" section; this naturally skips
    # baseline_queries.json (a list) and any unrelated JSON.
    if not isinstance(data, dict) or "aggregates" not in data:
        return None

    meta = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    test_id = meta.get("test_id") or _TS_SUFFIX.sub("", path.stem)
    ctx = _first_context(data)

    return {
        "path": path,
        "test_id": test_id,
        "timestamp": meta.get("timestamp", ""),
        "config": {key: _resolve(meta, ctx, key) for _, key in CONFIG_COLUMNS},
        "aggregates": data.get("aggregates", {}),
        "n_successful": meta.get("n_successful"),
        "n_total": meta.get("n_queries"),
    }


def _first_context(data: dict) -> dict:
    queries = data.get("queries", []) or []
    for q in queries:
        if q.get("status") == "ok" and q.get("context"):
            return q["context"]
    for q in queries:  # fall back to any context present
        if q.get("context"):
            return q["context"]
    return {}


def _resolve(meta: dict, ctx: dict, key: str):
    """Prefer an explicit metadata value, then the response context."""
    val = meta.get(key)
    if val in (None, ""):
        val = ctx.get(key)
    return val if val not in (None, "") else None


# ── Formatting ───────────────────────────────────────────────────────────────

def fmt_ms(value) -> str:
    return str(round(value)) if isinstance(value, (int, float)) else "—"


def agg(row: dict, stage: str, stat: str):
    s = row["aggregates"].get(stage)
    return s.get(stat) if isinstance(s, dict) else None


def fmt_p50_p95(row: dict, stage: str) -> str:
    p50, p95 = agg(row, stage, "p50"), agg(row, stage, "p95")
    if p50 is None and p95 is None:
        return "—"
    return f"{fmt_ms(p50)} / {fmt_ms(p95)}"


def fmt_cfg(value) -> str:
    return str(value) if value not in (None, "") else "—"


def delta_cell(row: dict, baseline: dict | None) -> str:
    if baseline is None:
        return "—"
    if row["path"] == baseline["path"]:
        return "baseline (T1)"
    cur, base = agg(row, "server_total_ms", "p50"), agg(baseline, "server_total_ms", "p50")
    if not isinstance(cur, (int, float)) or not isinstance(base, (int, float)) or base == 0:
        return "—"
    d = cur - base
    return f"{d:+.0f} ms ({d / base * 100:+.0f}%)"


# ── Ordering + baseline selection ────────────────────────────────────────────

def order_key(row: dict):
    tid = row["test_id"]
    rank = TEST_ORDER.index(tid) if tid in TEST_ORDER else len(TEST_ORDER)
    return (rank, tid)


def pick_baseline(rows: list[dict], explicit: str | None) -> dict | None:
    if explicit:
        return next((r for r in rows if r["test_id"] == explicit), None)
    # Auto: the most recent run whose test_id starts with "baseline".
    candidates = [r for r in rows if r["test_id"].lower().startswith("baseline")]
    if candidates:
        return max(candidates, key=lambda r: (r["timestamp"], r["test_id"]))
    return rows[0] if rows else None


# ── Rendering ────────────────────────────────────────────────────────────────

def build_table(rows: list[dict], baseline: dict | None) -> str:
    headers = (
        ["Test"]
        + [label for label, _ in CONFIG_COLUMNS]
        + ["server_total p50/p95", "embed p50", "dense p50", "rerank p50",
           "e2e p50", "Δ vs T1 (server_total p50)", "Runs"]
    )
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cfg = row["config"]
        n_ok, n_tot = row["n_successful"], row["n_total"]
        quality = f"{n_ok}/{n_tot}" if n_ok is not None and n_tot is not None else "—"
        cells = (
            [row["test_id"]]
            + [fmt_cfg(cfg.get(key)) for _, key in CONFIG_COLUMNS]
            + [
                fmt_p50_p95(row, "server_total_ms"),
                fmt_ms(agg(row, "embed_ms", "p50")),
                fmt_ms(agg(row, "dense_search_ms", "p50")),
                fmt_ms(agg(row, "rerank_ms", "p50")),
                fmt_ms(agg(row, "e2e_ms", "p50")),
                delta_cell(row, baseline),
                quality,
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(rows: list[dict], baseline: dict | None) -> str:
    parts = [build_table(rows, baseline)]
    base_id = baseline["test_id"] if baseline else "none"
    parts.append("")
    parts.append(f"_Baseline (T1) = `{base_id}`. Stage timings in ms (p50 unless noted). "
                 f"`—` = field not present in that run's JSON._")
    return "\n".join(parts)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=HERE,
                        help=f"directory to scan for result JSONs (default: {HERE})")
    parser.add_argument("--baseline", default=None,
                        help="test_id to use as the T1 baseline for the Δ column "
                             "(default: most recent baseline_* run)")
    parser.add_argument("--md-file", action="store_true",
                        help=f"also write the table to {RESULTS_MD.name} (default: stdout only)")
    args = parser.parse_args()

    rows = [r for r in (load_result(p) for p in discover_files(args.dir)) if r]
    if not rows:
        print(f"No result files ({' / '.join(PATTERNS)}) found in {args.dir}")
        return

    rows.sort(key=order_key)
    baseline = pick_baseline(rows, args.baseline)
    table = render(rows, baseline)
    print(table)

    if args.md_file:
        out = args.dir / RESULTS_MD.name
        generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        header = (
            "# Latency results\n\n"
            f"<!-- AUTO-GENERATED from {' / '.join(PATTERNS)} by summarize.py on "
            f"{generated}. Do not edit manually; re-run `python "
            "evaluation/latency/summarize.py --md-file`. -->\n\n"
            "> Auto-generated from the latency result JSONs — **do not edit manually.**\n\n"
        )
        out.write_text(header + table + "\n")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
