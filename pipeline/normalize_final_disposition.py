"""
Normalize the 151 unique final_disposition values into condensed canonical labels.
Appends a mapping section + condensed value_counts to reports/final_disposition_stats.txt
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lib.supabase_client import get_client

OUT_FILE = Path(__file__).resolve().parent.parent / "reports" / "final_disposition_stats.txt"

# ── Canonical sort order ──────────────────────────────────────────────────────
CANON_ORDER = ["Affirmed", "Denied", "Dismissed", "Granted", "Remanded",
               "Reversed", "Vacated"]

# ── Special-case overrides (checked before keyword extraction) ────────────────
OVERRIDES: dict[str, str] = {
    "(no evidence)":                                                    "Not Determined",
    "Not determined":                                                   "Not Determined",
    "Not determined.":                                                  "Not Determined",
    "Pending":                                                          "Pending",
    "Stayed":                                                           "Stayed",
    "Abeyance":                                                         "Abeyance",
    "Certified questions to Arizona Supreme Court":                     "Certified Question",
    "Rehearing en banc ordered":                                        "Reheard En Banc",
    "Reheard en banc":                                                  "Reheard En Banc",
    "Petition for Rehearing Denied as Moot":                            "Denied",
    "Conviction affirmed":                                              "Affirmed",
    # Stay-related — the stay itself being denied/granted is still a denial/grant
    "Stay of injunction denied":                                        "Denied",
    "Partial grant and partial denial of the Government's request for a stay": "Denied; Granted",
    # Free-text multi-party outcomes
    "Orozco-Lopez's petition is granted and remanded, while Gonzalez's is denied.":
        "Denied; Granted; Remanded",
    "Remanded for a new hearing as to Rivera-Recinos and Denied as to Recinos-Ulloa":
        "Denied; Remanded",
    "Petition denied in 23-358, and petition granted in 23-855":        "Denied; Granted",
    "Denied as to Zuliema Dolores Guerrero-Esperanza, Granted as to Daniel Enrique Vasquez-Guerrero; Remanded":
        "Denied; Granted; Remanded",
    # Verbose remand variants → just Remanded
    "Remanded for further proceedings":                                 "Remanded",
    "Remanded with instructions":                                       "Remanded",
    "Remanded with instructions to terminate proceedings":              "Remanded",
    "Remanded with instructions to grant CAT relief":                   "Remanded",
}

# ── Keyword → canonical term mapping ─────────────────────────────────────────
KEYWORD_PATTERNS: list[tuple[str, str]] = [
    (r"\baffirm",   "Affirmed"),
    (r"\bdeni",     "Denied"),
    (r"\bdismiss",  "Dismissed"),
    (r"\bgrant",    "Granted"),
    (r"\bremand",   "Remanded"),
    (r"\brevers",   "Reversed"),
    (r"\bvacat",    "Vacated"),
]


def canonicalize(raw: str) -> str:
    """Return a condensed canonical label for a raw final_disposition value."""
    stripped = raw.strip()

    # 1. Exact override lookup
    if stripped in OVERRIDES:
        return OVERRIDES[stripped]

    # 2. Keyword extraction
    found: set[str] = set()
    lower = stripped.lower()
    for pattern, label in KEYWORD_PATTERNS:
        if re.search(pattern, lower):
            found.add(label)

    if found:
        # Sort by canonical order, then alphabetically for any unknowns
        ordered = sorted(found, key=lambda x: (CANON_ORDER.index(x) if x in CANON_ORDER else 99, x))
        return "; ".join(ordered)

    # 3. Fallback — preserve original, flag it
    return f"Other: {stripped}"


def main() -> None:
    client = get_client()

    # Fetch all rows
    PAGE = 1000
    rows, offset = [], 0
    while True:
        batch = client.table("asylum_cases").select("final_disposition").range(offset, offset + PAGE - 1).execute().data
        if not batch:
            break
        rows.extend(batch)
        offset += PAGE

    df = pd.DataFrame(rows)
    col = df["final_disposition"]

    # Build mapping: original → canonical
    unique_vals = col.value_counts(dropna=False)
    mapping = {val: canonicalize(str(val)) for val in unique_vals.index}

    # Apply to full column
    col_canon = col.map(mapping)
    canon_vc = col_canon.value_counts(dropna=False)

    lines: list[str] = []

    # ── Section 1: original → canonical mapping table ────────────────────────
    lines.append("")
    lines.append("=" * 100)
    lines.append("NORMALIZATION MAPPING  (original value → canonical label)")
    lines.append("=" * 100)
    lines.append(f"{'#':<5} {'Original Value':<80} {'Count':>6}   {'Canonical Label'}")
    lines.append("-" * 130)
    for i, (orig, cnt) in enumerate(unique_vals.items(), 1):
        canon = mapping[orig]
        orig_str = str(orig) if orig is not None else "<NaN>"
        lines.append(f"{i:<5} {orig_str:<80} {cnt:>6,}   {canon}")

    # ── Section 2: condensed value_counts ────────────────────────────────────
    lines.append("")
    lines.append("=" * 100)
    lines.append(f"CONDENSED value_counts  ({len(canon_vc)} canonical labels from {len(unique_vals)} originals)")
    lines.append("=" * 100)
    lines.append(f"{'Canonical Label':<50} {'Count':>8}  {'Pct':>7}")
    lines.append("-" * 75)
    total = len(col_canon)
    for val, cnt in canon_vc.items():
        pct = cnt / total * 100
        lines.append(f"{str(val):<50} {cnt:>8,}  {pct:>6.2f}%")

    text = "\n".join(lines)
    with open(OUT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"Appended to → {OUT_FILE}")
    print(f"Original unique values : {len(unique_vals)}")
    print(f"Canonical labels       : {len(canon_vc)}")


if __name__ == "__main__":
    main()
