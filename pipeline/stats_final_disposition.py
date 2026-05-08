"""
Generate basic statistics on the final_disposition column from asylum_cases.

Outputs to reports/final_disposition_stats.txt
"""

import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lib.supabase_client import get_client

OUT_DIR = Path(__file__).resolve().parent.parent / "reports"
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "final_disposition_stats.txt"


def main() -> None:
    client = get_client()

    # Fetch only the final_disposition column — paginate to get all rows
    PAGE = 1000
    rows = []
    offset = 0
    while True:
        resp = (
            client.table("asylum_cases")
            .select("final_disposition")
            .range(offset, offset + PAGE - 1)
            .execute()
        )
        batch = resp.data
        if not batch:
            break
        rows.extend(batch)
        offset += PAGE

    df = pd.DataFrame(rows)
    col = df["final_disposition"]

    lines: list[str] = []

    # ── .info() equivalent ───────────────────────────────────────────────────
    buf = StringIO()
    df.info(buf=buf)
    lines.append("=" * 70)
    lines.append("df.info()")
    lines.append("=" * 70)
    lines.append(buf.getvalue())

    # ── .describe() ──────────────────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("col.describe()")
    lines.append("=" * 70)
    lines.append(str(col.describe()))
    lines.append("")

    # ── Null / missing audit ──────────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("Null / empty audit")
    lines.append("=" * 70)
    lines.append(f"Total rows          : {len(col):,}")
    lines.append(f"Null values         : {col.isna().sum():,}")
    lines.append(f"Empty-string values : {(col == '').sum():,}")
    lines.append(f"'Not determined'    : {(col == 'Not determined').sum():,}")
    lines.append(f"Unique values       : {col.nunique():,}")
    lines.append("")

    # ── Full value_counts ─────────────────────────────────────────────────────
    vc = col.value_counts(dropna=False)
    lines.append("=" * 70)
    lines.append(f"value_counts (all {len(vc):,} unique values, sorted by frequency)")
    lines.append("=" * 70)
    lines.append(f"{'Value':<80} {'Count':>8}  {'Pct':>7}")
    lines.append("-" * 100)
    total = len(col)
    for val, cnt in vc.items():
        label = str(val) if val is not None else "<NaN>"
        pct = cnt / total * 100
        lines.append(f"{label:<80} {cnt:>8,}  {pct:>6.2f}%")
    lines.append("")

    text = "\n".join(lines)
    OUT_FILE.write_text(text, encoding="utf-8")
    print(f"Saved → {OUT_FILE}")
    print(f"Rows fetched: {len(col):,}  |  Unique values: {col.nunique():,}")


if __name__ == "__main__":
    main()
