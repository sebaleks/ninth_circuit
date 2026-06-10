#!/usr/bin/env python3
"""Latency-reduction charts (T1 NIM baseline -> deployed no-NIM T_optimized).

Minimal-text, clean aesthetic: no y-axis clutter, values on bars, short labels.
Reads the committed result JSONs. Output: evaluation/latency/latency_reduction.png
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.rcParams.update({
    "figure.facecolor": "#f7f7f7", "axes.facecolor": "#f7f7f7",
    "axes.labelcolor": "#373737", "text.color": "#373737",
    "xtick.color": "#373737", "ytick.color": "#373737",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.grid": False, "font.family": "DejaVu Sans",
})
SLOW = "#b2182b"     # NIM baseline / removed
FAST = "#2a9d8f"     # deployed no-NIM stack
INK = "#373737"
MUTE = "#9aa0a6"


def load(p):
    f = sorted(glob.glob(str(HERE / p)))
    return json.loads(Path(f[-1]).read_text()) if f else {}


def g(d, stage, stat="p50"):
    s = d.get("aggregates", {}).get(stage)
    return (s.get(stat) if isinstance(s, dict) else None) or 0


t1, topt = load("T1_baseline_api_direct.json"), load("T_optimized_baseline_api_direct_*.json")
t1v, toptv = load("T1_baseline_vercel_proxy_*.json"), load("T_optimized_baseline_vercel_proxy_*.json")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
for a in ax:
    a.set_yticks([]); a.tick_params(length=0)


def label(a, bars, vals, fmt="{:,.0f}"):
    top = max(vals)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v + top * 0.02, fmt.format(v),
               ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)


# ── 1. server-side (headline) ──
s1, s2 = g(t1, "server_total_ms"), g(topt, "server_total_ms")
b = ax[0].bar(["T1", "T-optimized"], [s1, s2], color=[SLOW, FAST], width=0.5)
label(ax[0], b, [s1, s2], "{:,.0f} ms")
ax[0].set_ylim(0, s1 * 1.25)
ax[0].set_title("Server-side  (p50)", fontsize=13, color=INK, pad=14)
ax[0].annotate(f"{s1/s2:.0f}× faster", xy=(0.5, s1 * 0.62), ha="center",
               fontsize=15, fontweight="bold", color=SLOW)

# ── 2. per-stage (the why) ──
order = [("embed", "embed_ms", SLOW), ("rerank", "rerank_ms", "#ef8a62"),
         ("dense", "dense_search_ms", FAST), ("bm25", "bm25_ms", "#9fd3cc")]
for i, d in enumerate([t1, topt]):
    bottom = 0
    for name, key, col in order:
        v = g(d, key)
        ax[1].bar(i, v, bottom=bottom, color=col, width=0.5)
        if v > 120:  # only label the dominant slices
            ax[1].text(i, bottom + v / 2, name, ha="center", va="center", color="white", fontsize=10)
        bottom += v
    ax[1].text(i, bottom + s1 * 0.02, f"{bottom:,.0f} ms", ha="center", fontsize=12, fontweight="bold", color=INK)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["T1", "T-optimized"])
ax[1].set_ylim(0, s1 * 1.25)
ax[1].set_title("Per-stage  (NIM embed + rerank removed)", fontsize=13, color=INK, pad=14)

# ── 3. end-to-end (user-visible) ──
e1 = [g(t1, "e2e_ms"), g(t1v, "e2e_ms")]
e2 = [g(topt, "e2e_ms"), g(toptv, "e2e_ms")]
b1 = ax[2].bar([-0.21, 0.79], e1, width=0.4, color=SLOW)
b2 = ax[2].bar([0.21, 1.21], e2, width=0.4, color=FAST)
for xs, vs in [([-0.21, 0.79], e1), ([0.21, 1.21], e2)]:
    for x, v in zip(xs, vs):
        ax[2].text(x, v + max(e1) * 0.02, f"{v:,.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["direct", "proxy"])
ax[2].set_ylim(0, max(e1) * 1.2)
ax[2].set_title("End-to-end  (p50, ms)", fontsize=13, color=INK, pad=14)
ax[2].legend([b1, b2], ["T1", "T-optimized"], frameon=False, fontsize=10, loc="upper right")

plt.tight_layout()
out = HERE / "latency_reduction.png"
plt.savefig(out, dpi=130, facecolor="#f7f7f7", bbox_inches="tight")
print(f"wrote {out}")
print(f"server p50 {s1:.0f}->{s2:.0f}ms ({s1/s2:.0f}x); e2e direct {e1[0]:.0f}->{e2[0]:.0f}, proxy {e1[1]:.0f}->{e2[1]:.0f}")
