import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": [s]})
code = lambda s: cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s]})

md("""# Asylum RAG — latency results

Reproducible charts for the latency-reduction work: **T1** (NIM 2048 + rerank + FAISS baseline) → **T_optimized** (local ONNX e5 + Qdrant + RRF, no NIM — deployed). All charts are generated from the committed result JSONs in `evaluation/latency/`; re-run after any new test and they update automatically.""")

code('''import json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

DIR = next((p for p in [Path("evaluation/latency"), Path(".")]
            if (p / "T1_baseline_api_direct.json").exists()), Path("."))
print("results dir:", DIR.resolve())

# pleasant palette
BG, INK = "#faf8f5", "#3d405b"
SLOW, FAST = "#e07a5f", "#3d9991"                 # baseline (NIM) vs optimized
STAGE = {"embed": "#e07a5f", "rerank": "#f2cc8f", "dense": "#3d9991", "bm25": "#a8dadc"}
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.spines.bottom": False, "axes.grid": False, "font.family": "DejaVu Sans", "font.size": 11,
})

def load(pat):
    f = sorted(glob.glob(str(DIR / pat)))
    return json.loads(Path(f[-1]).read_text()) if f else {}

def agg(d, stage, stat="p50"):
    s = d.get("aggregates", {}).get(stage)
    return (s.get(stat) if isinstance(s, dict) else None) or 0''')

md("""## 1. Headline — the latency reduction
Server-side **~38×**, end-to-end **~7×**, by removing the two NIM API round-trips (embed + rerank).""")

code('''t1, topt = load("T1_baseline_api_direct.json"), load("T_optimized_baseline_api_direct_*.json")
t1v, toptv = load("T1_baseline_vercel_proxy_*.json"), load("T_optimized_baseline_vercel_proxy_*.json")
s1, s2 = agg(t1, "server_total_ms"), agg(topt, "server_total_ms")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
for a in ax: a.set_yticks([]); a.tick_params(length=0)

b = ax[0].bar(["T1", "T-optimized"], [s1, s2], color=[SLOW, FAST], width=0.5)
for bar, v in zip(b, [s1, s2]):
    ax[0].text(bar.get_x()+bar.get_width()/2, v+s1*0.02, f"{v:,.0f} ms", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax[0].set_ylim(0, s1*1.25); ax[0].set_title("Server-side  (p50)", fontsize=13, pad=14)
ax[0].annotate(f"{s1/s2:.0f}× faster", xy=(0.5, s1*0.6), ha="center", fontsize=15, fontweight="bold", color=SLOW)

for i, d in enumerate([t1, topt]):
    bottom = 0
    for name, key in [("embed","embed_ms"),("rerank","rerank_ms"),("dense","dense_search_ms"),("bm25","bm25_ms")]:
        v = agg(d, key); ax[1].bar(i, v, bottom=bottom, color=STAGE[name], width=0.5)
        if v > 120: ax[1].text(i, bottom+v/2, name, ha="center", va="center", color="white", fontsize=10)
        bottom += v
    ax[1].text(i, bottom+s1*0.02, f"{bottom:,.0f} ms", ha="center", fontsize=12, fontweight="bold")
ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["T1","T-optimized"]); ax[1].set_ylim(0, s1*1.25)
ax[1].set_title("Per-stage  (NIM embed + rerank removed)", fontsize=13, pad=14)

e1, e2 = [agg(t1,"e2e_ms"), agg(t1v,"e2e_ms")], [agg(topt,"e2e_ms"), agg(toptv,"e2e_ms")]
bb1 = ax[2].bar([-0.21, 0.79], e1, width=0.4, color=SLOW)
bb2 = ax[2].bar([0.21, 1.21], e2, width=0.4, color=FAST)
for xs, vs in [([-0.21,0.79], e1), ([0.21,1.21], e2)]:
    for x, v in zip(xs, vs): ax[2].text(x, v+max(e1)*0.02, f"{v:,.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax[2].set_xticks([0,1]); ax[2].set_xticklabels(["direct","proxy"]); ax[2].set_ylim(0, max(e1)*1.2)
ax[2].set_title("End-to-end  (p50, ms)", fontsize=13, pad=14)
ax[2].legend([bb1, bb2], ["T1","T-optimized"], frameon=False, fontsize=10, loc="upper right")

plt.tight_layout(); plt.savefig(DIR/"latency_reduction.png", dpi=130, bbox_inches="tight"); plt.show()''')

md("""## 2. All tests — data-driven
Loads **every** `T*_baseline_*.json` and tabulates + charts them. Add a new run and re-run this section; it appears automatically.""")

code('''rows = []
for f in sorted(glob.glob(str(DIR / "T*_baseline_*.json"))):
    d = json.loads(Path(f).read_text()); m = d.get("metadata", {})
    rows.append({"test": m.get("test_id"), "path": m.get("label"),
                 "server_p50": agg(d, "server_total_ms"), "server_p95": agg(d, "server_total_ms", "p95"),
                 "e2e_p50": agg(d, "e2e_ms"), "embed": agg(d, "embed_ms"), "rerank": agg(d, "rerank_ms"),
                 "dense": agg(d, "dense_search_ms"), "bm25": agg(d, "bm25_ms")})
df = pd.DataFrame(rows).sort_values(["test", "path"]).reset_index(drop=True)
df''')

code('''labels = [f"{r.test}\\n{r.path}" for r in df.itertuples()]
x = np.arange(len(df))
fig, a = plt.subplots(figsize=(max(9, 1.5*len(df)), 5))
a.bar(x-0.2, df.server_p50, width=0.4, color=FAST, label="server p50")
a.bar(x+0.2, df.e2e_p50,   width=0.4, color=SLOW, label="e2e p50")
for i, (s, e) in enumerate(zip(df.server_p50, df.e2e_p50)):
    a.text(i-0.2, s, f"{s:,.0f}", ha="center", va="bottom", fontsize=9)
    a.text(i+0.2, e, f"{e:,.0f}", ha="center", va="bottom", fontsize=9)
a.set_xticks(x); a.set_xticklabels(labels, fontsize=9); a.set_yticks([])
a.set_title("All latency runs — server vs end-to-end (p50, ms)", fontsize=13, pad=14)
a.legend(frameon=False); plt.tight_layout(); plt.show()''')

md("""## 3. Design choice — fusion & reranker  *(add-on)*
RRF was chosen because it's **free**; the cross-encoder reranker would have cost ~660 ms.""")

code('''variants = [("RRF\\n(deployed)", "T_optimized_baseline_api_direct_*.json"),
            ("blend", "T_optimized_blend_baseline_api_direct_*.json"),
            ("blend +\\nreranker", "T_optimized_blend_rerank_baseline_api_direct_*.json")]
names = [n for n, _ in variants]; vals = [agg(load(p), "server_total_ms") for _, p in variants]
fig, a = plt.subplots(figsize=(7, 4.5))
b = a.bar(names, vals, color=[FAST, FAST, SLOW], width=0.55)
for bar, v in zip(b, vals): a.text(bar.get_x()+bar.get_width()/2, v, f"{v:,.0f} ms", ha="center", va="bottom", fontweight="bold")
a.set_yticks([]); a.set_ylim(0, max(vals)*1.18)
a.set_title("Server p50 by fusion/reranker — RRF is free, reranker ~+660 ms", fontsize=12, pad=14)
plt.tight_layout(); plt.show()''')

md("""## 4. Why NIM was the bottleneck — T2 & T3 probes  *(add-on)*""")

code('''t2 = json.loads((DIR/"T2_nim_dimension_probe.json").read_text())
t3 = json.loads((DIR/"T3_faiss_vs_qdrant_dense.json").read_text())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

dims = list(t2["by_dim"].keys()); p50 = [t2["by_dim"][k]["p50_ms"] for k in dims]
b = ax1.bar(dims, p50, color=SLOW, width=0.55)
for bar, v in zip(b, p50): ax1.text(bar.get_x()+bar.get_width()/2, v, f"{v/1000:.1f}s", ha="center", va="bottom", fontweight="bold")
ax1.set_yticks([]); ax1.set_xlabel("NIM embed dimension"); ax1.set_ylim(0, max(p50)*1.18)
ax1.set_title("T2 · NIM embed p50 — dimension-neutral (~15 s)", fontsize=12, pad=12)

labs = ["FAISS\\n(in-process)", "Qdrant\\n(network)"]; vals = [t3["faiss_ms"]["p50"], t3["qdrant_ms"]["p50"]]
b = ax2.bar(labs, vals, color=[FAST, SLOW], width=0.55); ax2.set_yscale("log")
for bar, v in zip(b, vals): ax2.text(bar.get_x()+bar.get_width()/2, v, f"{v:.2f} ms", ha="center", va="bottom", fontweight="bold")
ax2.set_title("T3 · dense-search p50 (log scale)", fontsize=12, pad=12)
plt.tight_layout(); plt.show()''')

md("""## 5. Export standalone slide panels  *(add-on)*
Saves the two key visuals as separate PNGs for slides.""")

code('''# server-side standalone
fig, a = plt.subplots(figsize=(5, 4.6))
b = a.bar(["T1", "T-optimized"], [s1, s2], color=[SLOW, FAST], width=0.5)
for bar, v in zip(b, [s1, s2]): a.text(bar.get_x()+bar.get_width()/2, v+s1*0.02, f"{v:,.0f} ms", ha="center", va="bottom", fontsize=13, fontweight="bold")
a.set_yticks([]); a.set_ylim(0, s1*1.25); a.set_title("Server-side latency (p50)", fontsize=13, pad=12)
a.annotate(f"{s1/s2:.0f}× faster", xy=(0.5, s1*0.6), ha="center", fontsize=16, fontweight="bold", color=SLOW)
plt.tight_layout(); plt.savefig(DIR/"panel_server.png", dpi=130, bbox_inches="tight"); plt.show()

# end-to-end standalone
fig, a = plt.subplots(figsize=(5.5, 4.6))
a.bar([-0.21,0.79], e1, width=0.4, color=SLOW, label="T1")
a.bar([0.21,1.21], e2, width=0.4, color=FAST, label="T-optimized")
for xs, vs in [([-0.21,0.79], e1), ([0.21,1.21], e2)]:
    for x, v in zip(xs, vs): a.text(x, v+max(e1)*0.02, f"{v:,.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
a.set_xticks([0,1]); a.set_xticklabels(["direct","proxy"]); a.set_yticks([]); a.set_ylim(0, max(e1)*1.2)
a.set_title("End-to-end latency (p50, ms)", fontsize=13, pad=12); a.legend(frameon=False)
plt.tight_layout(); plt.savefig(DIR/"panel_e2e.png", dpi=130, bbox_inches="tight"); plt.show()
print("saved panel_server.png, panel_e2e.png")''')

# ════════════════════════ Part II — accuracy / retrieval quality ════════════════════════
md("""# Part II — accuracy / retrieval-quality findings

*Caveats:* on the **30-case sample / 6-query thematic gold** — a signal, not a verdict. The product is **case retrieval, not Q&A**, so abstention is secondary (a search tool ranks; the lawyer judges relevance). Full reasoning, every claim traced to an artifact: `RETRIEVAL_FINDINGS.md`.""")

md("""## A1. Anisotropy → the dense-cosine abstention gate is dead on e5

e5-small-v2 is severely **anisotropic** — every query's top cosine lands in ~0.78–0.82, so "is this in-corpus?" can't be thresholded (FGM scores the same as a real claim). NIM nemotron is discriminative: in-corpus / absent-harm / off-domain separate, and the 0.15 refuse-threshold sits in the gap.""")

code('''# dense max-cosine by query type (sources: embedder_sweep + NIM probe; RETRIEVAL_FINDINGS.md 1c/3)
cats = ["in-corpus", "FGM/absent", "off-domain"]
e5  = [0.824, 0.819, 0.779]      # e5-small-v2: flat -> no gate possible
nim = [0.217, 0.122, 0.054]      # NIM nemotron-1b: separates
fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, vals, title, sep in [(axa, e5, "e5-small-v2  (anisotropic)", None),
                             (axb, nim, "NIM nemotron-1b  (discriminative)", 0.15)]:
    b = ax.bar(cats, vals, color=[FAST, SLOW, "#bdbdbd"], width=0.6)
    for bar, v in zip(b, vals): ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
    if sep:
        ax.axhline(sep, ls="--", color=INK, lw=1.2)
        ax.text(2.45, sep+0.006, "refuse < 0.15", ha="right", va="bottom", fontsize=9)
    ax.set_yticks([]); ax.set_ylim(0, max(vals)*1.3); ax.set_title(title, fontsize=12, pad=12)
axa.text(1, 0.86, "FGM ~= in-corpus -> no threshold works", ha="center", fontsize=9, color=SLOW)
fig.suptitle("Abstention: dense max-cosine by query type", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()''')

md("""## A2. Embedder choice — retrieval quality (abstention needs NIM regardless)

**gte-base is the best *local* retriever (7/7 core).** But no local embedder — small or mid-size — gives a usable abstention gate; only NIM does, at ~15 s/query. So embedder choice trades on *retrieval*; abstention is a separate axis no local model unlocks.""")

code('''# core gold recovered (/7), all swept embedders (sources: embedder_sweep_results.md + _midsize_results.md)
ret = {"gte-base":7, "e5-small":6, "NIM-2048":6, "e5-large":5,
       "e5-base":4, "bge-base":4, "gte-small":4, "bge-small":4, "MiniLM-L6":3}
items = sorted(ret.items(), key=lambda kv: -kv[1]); names=[k for k,_ in items]; vals=[v for _,v in items]
cols = [FAST if k=="gte-base" else (SLOW if k.startswith("NIM") else "#cfcfcf") for k in names]
fig, a = plt.subplots(figsize=(10, 4.6))
b = a.bar(names, vals, color=cols, width=0.62)
for bar, v in zip(b, vals): a.text(bar.get_x()+bar.get_width()/2, v, f"{v}/7", ha="center", va="bottom", fontweight="bold")
a.set_yticks([]); a.set_ylim(0, 8); a.set_title("Retrieval — core gold recovered (/7), 6-query thematic set", fontsize=12, pad=12)
plt.setp(a.get_xticklabels(), rotation=20, ha="right", fontsize=9)
a.text(-0.45, 7.5, "gte-base = best local", color=FAST, fontsize=10, fontweight="bold")
plt.tight_layout(); plt.show()''')

md("""## A3. The reranker did **not** help

A `gte-reranker-modernbert-base` (149 M) cross-encoder stacked on RRF, tested against the gold:
- **No retrieval gain** (core recovery dropped, in fact) and it **broke the citation sentinel** — `21-70493` fell *out* of the top-5 (was rank 5).
- It **failed its primary target** (DV claim-agency: the perpetrator `21-70493` still outranked the victim).
- And it cost **~2 s/query** on a model that **OOMs the 512 MB free tier**. RRF (free) wins outright.""")

code('''TH = DIR.parent / "thematic"
def thload(n):
    p = TH / n
    return json.loads(p.read_text()) if p.exists() else None
rrf, gte = thload("rrf_beta0_results.json"), thload("gte_rrf_beta0_results.json")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
if rrf and gte:
    den = sum(r["core_total"] for r in rrf["rows"])
    cr = [sum(r["core_recovered"] for r in rrf["rows"]), sum(r["core_recovered"] for r in gte["rows"])]
    b = a1.bar(["RRF", "RRF + gte"], cr, color=[FAST, SLOW], width=0.5)
    for bar, v in zip(b, cr): a1.text(bar.get_x()+bar.get_width()/2, v, f"{v}/{den}", ha="center", va="bottom", fontweight="bold")
    a1.set_ylim(0, den+1)
a1.set_yticks([]); a1.set_title("Core gold recovered  (gte: no gain, lost the citation)", fontsize=11, pad=12)
b = a2.bar(["RRF", "RRF + gte"], [0, 2016], color=[FAST, SLOW], width=0.5)   # gte rerank ~2016ms/query (pool=25)
for bar, v in zip(b, [0, 2016]): a2.text(bar.get_x()+bar.get_width()/2, v, ("free" if v==0 else f"+{v:,} ms"), ha="center", va="bottom", fontweight="bold")
a2.set_yticks([]); a2.set_ylim(0, 2400); a2.set_title("Added latency / query  (gte 149M, OOMs 512 MB)", fontsize=11, pad=12)
plt.tight_layout(); plt.show()''')

md("""## A4. Other notable findings

- **Attractor cases.** A few long *policy* opinions (East Bay Sanctuary, Immigrant Defenders) are term-dense across many harms and dominate any harm query — they buried real individual-claim cases until **RRF** (rank-based fusion) neutralized their magnitude advantage.
- **Citation fragility.** Docket lookups are phrasing-sensitive: `21-70493` ranked 3 / 5 / *absent* depending on wording. A deterministic **docket exact-match bypass** is the targeted fix.
- **Content hygiene.** A signature/boilerplate chunk hub-collapsed *above* real content (cosine 0.83); a `content_words < 12` filter cleanly removed it (1 chunk on 30) — shipped in the clean collection.
- **DV claim-agency.** "Domestic violence" surfaced the *perpetrator's* removal case over the *victim's* claim — a polarity failure no fusion/reranker fixed; needs claim-agency metadata.

Full chain of evidence: **`RETRIEVAL_FINDINGS.md`**.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path("evaluation/latency/results.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")
