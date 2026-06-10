#!/usr/bin/env python3
"""Summarize latency result JSONs into one markdown comparison table.

Scans evaluation/latency/ for result files (the T-series and any test_*.json),
pulls each run's configuration + headline metrics, and renders a table sorted by
test id then path. A Δ column reports each run's e2e p50 relative to the
T1 api_direct baseline.

Reads the JSON shape written by run_baseline.py:
    {"metadata": {...}, "aggregates": {stage: {p50,p95,mean,n}}, "queries":[...]}
Per-run config (embedder_name, embedding_dim, vector_store, rerank_pool_size)
comes from the first successful query's `context`. Missing fields render as "—".

The measurement path is derived from the filename: "vercel_proxy", "api_direct",
else "other".

Usage:
    python evaluation/latency/summarize.py
    python evaluation/latency/summarize.py --include-stages
    python evaluation/latency/summarize.py --md-file            # -> RESULTS.md
    python evaluation/latency/summarize.py --md-file out.md
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
RESULTS_MD = HERE / "RESULTS.md"

# Generalized to cover every enumerated pattern (T1_baseline_api_direct*,
# T1_baseline_vercel_proxy*, T2_baseline_*, T3_*, T4_*) and any test_*.json.
PATTERNS = ["T*_*.json", "test_*.json"]

# Experiments that were intentionally NOT measured — recorded so a missing row in the
# table isn't a mystery. (The table is data-driven; parked experiments have no JSON.)
PARKED_NOTE = """## Parked experiments

**T2 — NIM 512-dim Matryoshka — parked, no baseline captured.**
- **NIM free-tier embed latency degraded to ~13 s** during the T2 probe — vs T1's clean
  NIM embed p50 **559 ms** (a ~23× spike). A clean baseline couldn't be taken, and this
  transient instability was a key reason for removing the NIM dependency entirely.
  *(Observed live during probing; never saved as a result JSON — hence no T2 row above.)*
- The dimension probe showed `dimensions=` is **latency-neutral** (512 ≈ 2048 ≈ none):
  Matryoshka is a **storage lever, not a latency lever**, so a T2 latency row would add nothing.

→ We went straight from **T1** (NIM: embed 559 ms + rerank 549 ms → server_total **1110 ms**)
to **T_optimized** (local ONNX e5 + Qdrant, no NIM → server_total **29 ms**)."""

# Extra stage columns shown only with --include-stages: (header, stage key).
STAGE_COLUMNS = [
    ("embed p50", "embed_ms"),
    ("dense p50", "dense_search_ms"),
    ("rerank p50", "rerank_ms"),
    ("bm25 p50", "bm25_ms"),
    ("net p50", "network_overhead_ms"),
]


# ── Extraction ───────────────────────────────────────────────────────────────

def test_id_of(name: str) -> str:
    # Filenames are "<test-id>_baseline_<label>_<ts>.json"; the test id is
    # whatever precedes "_baseline_" (covers T1, T2, T_optimized, …). Fall back
    # to the legacy "T<digits>" prefix, then the bare stem.
    if "_baseline_" in name:
        return name.split("_baseline_", 1)[0]
    m = re.match(r"^(T\d+)", name)
    return m.group(1) if m else Path(name).stem


def path_of(name: str) -> str:
    n = name.lower()
    if "vercel_proxy" in n:
        return "vercel_proxy"
    if "api_direct" in n:
        return "api_direct"
    return "other"


def first_context(data: dict) -> dict:
    queries = data.get("queries", []) or []
    for q in queries:
        if q.get("status") == "ok" and q.get("context"):
            return q["context"]
    for q in queries:
        if q.get("context"):
            return q["context"]
    return {}


def agg(data: dict, stage: str, stat: str):
    s = data.get("aggregates", {}).get(stage)
    return s.get(stat) if isinstance(s, dict) else None


def load_row(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict) or "aggregates" not in data:
        return None  # skips baseline_queries.json and any non-result JSON
    meta = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    ctx = first_context(data)
    return {
        "name": path.name,
        "link": f"evaluation/latency/{path.name}",
        "test_id": test_id_of(path.name),
        "path": path_of(path.name),
        "embedder": ctx.get("embedder_name"),
        "dim": ctx.get("embedding_dim"),
        "store": ctx.get("vector_store"),
        "rerank_pool": ctx.get("rerank_pool_size"),
        "server_p50": agg(data, "server_total_ms", "p50"),
        "server_p95": agg(data, "server_total_ms", "p95"),
        "e2e_p50": agg(data, "e2e_ms", "p50"),
        "stages": {key: agg(data, key, "p50") for _, key in STAGE_COLUMNS},
        "n_ok": meta.get("n_successful"),
        "n_total": meta.get("n_queries"),
        "timestamp": meta.get("timestamp", ""),
    }


def discover(directory: Path) -> list[dict]:
    seen: dict[str, Path] = {}
    for pattern in PATTERNS:
        for p in directory.glob(pattern):
            seen[p.name] = p
    rows = [r for r in (load_row(p) for p in seen.values()) if r]
    rows.sort(key=lambda r: (_tnum(r["test_id"]), r["test_id"], r["path"]))
    return rows


def _tnum(test_id: str) -> int:
    m = re.match(r"^T(\d+)$", test_id)
    return int(m.group(1)) if m else 9999


# ── Formatting ───────────────────────────────────────────────────────────────

def ms(v) -> str:
    return str(round(v)) if isinstance(v, (int, float)) else "—"


def cfg(v) -> str:
    return str(v) if v not in (None, "") else "—"


def server_cell(row: dict) -> str:
    if row["server_p50"] is None and row["server_p95"] is None:
        return "—"
    return f"{ms(row['server_p50'])} / {ms(row['server_p95'])}"


def find_baseline(rows: list[dict]) -> dict | None:
    return next((r for r in rows if r["test_id"] == "T1" and r["path"] == "api_direct"), None)


def delta_cell(row: dict, baseline: dict | None) -> str:
    if baseline is None:
        return "—"
    if row is baseline:
        return "(baseline)"
    cur, base = row["e2e_p50"], baseline["e2e_p50"]
    if not isinstance(cur, (int, float)) or not isinstance(base, (int, float)):
        return "—"
    return f"{cur - base:+.0f} ms"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def summary_line(rows: list[dict]) -> str:
    n = len(rows)
    n_tests = len({r["test_id"] for r in rows})
    valid = [r for r in rows if isinstance(r["e2e_p50"], (int, float))]
    if valid:
        best = min(valid, key=lambda r: r["e2e_p50"])
        best_str = (f"Best e2e p50: {round(best['e2e_p50'])} ms "
                    f"(Test {best['test_id']}, path {best['path']}).")
    else:
        best_str = "Best e2e p50: n/a."
    return f"{plural(n, 'measurement')} across {plural(n_tests, 'test')}. {best_str}"


def build_table(rows: list[dict], include_stages: bool) -> str:
    baseline = find_baseline(rows)
    headers = ["Test", "Path", "Embedder", "Dim", "Store", "server p50/p95 (ms)", "e2e p50 (ms)"]
    if include_stages:
        headers += [h for h, _ in STAGE_COLUMNS]
    headers += ["Δ vs T1 api_direct", "n_ok", "File"]

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        n_ok = (f"{row['n_ok']}/{row['n_total']}"
                if row["n_ok"] is not None and row["n_total"] is not None else "—")
        cells = [
            row["test_id"], row["path"], cfg(row["embedder"]), cfg(row["dim"]),
            cfg(row["store"]), server_cell(row), ms(row["e2e_p50"]),
        ]
        if include_stages:
            cells += [ms(row["stages"][key]) for _, key in STAGE_COLUMNS]
        cells += [delta_cell(row, baseline), n_ok, f"[{row['name']}]({row['link']})"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=HERE,
                        help=f"directory to scan (default: {HERE})")
    parser.add_argument("--md-file", nargs="?", const=str(RESULTS_MD), default=None,
                        metavar="PATH",
                        help=f"also write the table to a markdown file (default: {RESULTS_MD})")
    parser.add_argument("--include-stages", action="store_true",
                        help="add per-stage p50 columns (embed/dense/rerank/bm25/net)")
    args = parser.parse_args()

    rows = discover(args.dir)
    if not rows:
        print(f"No result files ({' / '.join(PATTERNS)}) found in {args.dir}")
        return

    summary = summary_line(rows)
    table = build_table(rows, args.include_stages)

    print(summary)
    print()
    print(table)
    print()
    print(PARKED_NOTE)

    if args.md_file:
        out = Path(args.md_file)
        generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out.write_text(
            "# Latency results\n\n"
            f"<!-- AUTO-GENERATED by summarize.py on {generated}. Do not edit by hand; "
            "re-run `python evaluation/latency/summarize.py --md-file`. -->\n\n"
            f"{summary}\n\n{table}\n\n{PARKED_NOTE}\n"
        )
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
