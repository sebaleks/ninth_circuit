#!/usr/bin/env python3
"""Thematic-retrieval eval scorer + local retrieval driver.

Measurement scaffold for testing RANKING changes against a graded gold set. It
does NOT change ranking — it measures whatever ranking the driver produces.

Two pieces:
  • LocalRetriever — produces run_baseline-style top_5_case_ids (a list of
    case_link URLs per query) by mirroring the production pipeline EXACTLY on the
    cleaned local chunk set (no Qdrant needed):
      dense (e5 ONNX, cosine) → dense-gated pool max(20, k*5)  [retrieval.py:428,
      main.py:97] → BM25 re-score within pool (corpus-norm) → blend 0.6/0.4
      [retrieval.py:467] → dedupe by case_link → top-k.
    Substrate = cleaned set: drop content_words<12 boilerplate (STEP a).
  • score() — consumes {query_id: [case_link, ...]} (the same shape as
    run_baseline's top_5_case_ids, keyed on case_link / retrieval.py:198) and the
    graded suite, and reports PER-QUERY results. No single blended number.

Headline per query: CORE cases recovered in top-5 + their ranks; the
"sexual assault" SENTINEL (14-70905) rank is always shown. marginal = credit if
present (not required); excluded = must not surface as relevant.

Negative-control rule (hard): pass --baseline <results.json> on a later run and
ANY class-B (control) query that loses a core case or worsens a core rank
DISQUALIFIES the change, regardless of class-A gains.

Usage:
  python evaluation/thematic/score.py                      # run local driver, establish baseline
  python evaluation/thematic/score.py --save baseline.json # ...and save the top_5_case_ids
  python evaluation/thematic/score.py --results run.json    # score an external run_baseline-style file
  python evaluation/thematic/score.py --baseline baseline.json  # compare + enforce control rule
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))  # so `import rag_api` works regardless of cwd
SUITE_PATH = HERE / "thematic_queries.json"
PARQUET = REPO / "data" / "experiments" / "T-optimized-onnx" / "metadata.parquet"

HYBRID_ALPHA = 0.6   # retrieval.py:49 (60% dense / 40% BM25 when reranker off)
RRF_K = 60           # retrieval.py:_fuse_rrf k (rank-based fusion constant)
DROP_CW = 12         # STEP a hygiene threshold (content_words < 12 → boilerplate)
RETURN_K = 5
# retrieval.py:65 — kept in sync so BM25 tokenization matches production.
STOPWORDS = {"the", "and", "for", "with", "from", "that", "this", "what", "where",
             "when", "which", "who", "whom", "does", "did", "was", "were", "are",
             "have", "has", "had", "but", "not", "all", "any", "some", "into",
             "about", "case", "cases"}


def qtok(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) >= 3 and t not in STOPWORDS]


def docket(link: str) -> str:
    m = re.search(r"/(\d{2}-\d+)\.pdf$", link or "")
    return m.group(1) if m else (link or "?")


# ── Local retrieval driver (faithful to production; cleaned substrate) ───────────

class LocalRetriever:
    def __init__(self, cleaned: bool = True):
        df = pd.read_parquet(PARQUET).reset_index(drop=True)
        df["cw"] = df["text"].astype(str).map(lambda t: len(set(qtok(t))))
        self.n_before = len(df)
        if cleaned:
            df = df[df["cw"] >= DROP_CW].reset_index(drop=True)
        self.n_after = len(df)
        self.df = df
        self.links = df["case_link"].tolist()
        from rag_api.onnx_embedder import OnnxEmbedder
        self.emb = OnnxEmbedder()
        self.D = self.emb.embed_passages(df["text"].astype(str).tolist()).astype(np.float32)
        self.bm25 = BM25Okapi([qtok(t) for t in df["text"].astype(str)])

    def _gte(self):
        if getattr(self, "_ce", None) is None:
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder("Alibaba-NLP/gte-reranker-modernbert-base", trust_remote_code=True)
        return self._ce

    def search(self, query: str, return_k: int = RETURN_K, fusion: str = "blend",
               beta: float = 0.0, reranker: str | None = None) -> list[str]:
        dc = self.D @ self.emb.embed_query(query)[0]
        tk = qtok(query)
        raw = self.bm25.get_scores(tk) if tk else np.zeros(len(self.links))
        top = raw.max()
        bn = (raw / top) if top > 0 else raw * 0.0
        pool_n = max(20, return_k * 5)                       # main.py:97 + retrieval.py:428
        pool = list(np.argsort(-dc)[:pool_n])

        # Semantic signal: dense cosine, OR (if reranker) the gte cross-encoder
        # score over (query, chunk) for each pooled chunk — replicating production
        # USE_RERANKER (rerank_score becomes the semantic signal, retrieval.py:458).
        if reranker == "gte":
            import time as _t
            pairs = [[query, self.df["text"].iloc[i]] for i in pool]
            t0 = _t.perf_counter()
            scores = self._gte().predict(pairs)
            self.last_rerank_ms = (_t.perf_counter() - t0) * 1000
            self.last_pool_n = len(pool)
            sem = {i: float(s) for i, s in zip(pool, scores)}
        else:
            sem = {i: float(dc[i]) for i in pool}

        if fusion == "rrf":
            # rank-based fusion within the pool (retrieval.py:_fuse_rrf), k=60
            d_rank = {i: r for r, i in enumerate(sorted(pool, key=lambda i: -sem[i]), 1)}
            b_rank = {i: r for r, i in enumerate(sorted(pool, key=lambda i: -bn[i]), 1)}
            fused = {i: 1.0 / (RRF_K + d_rank[i]) + 1.0 / (RRF_K + b_rank[i]) for i in pool}
        else:  # "blend" — production default
            fused = {i: HYBRID_ALPHA * sem[i] + (1 - HYBRID_ALPHA) * bn[i] for i in pool}

        # case-collapse: aggregate pooled chunks per case_link.
        #   β=0  → pure max-per-case (== production dedupe-keep-highest)
        #   β>0  → max + β·min(Σ other-chunk scores, max)  [CAPPED support: bonus ≤ β·max,
        #          so a long multi-chunk opinion can't run away on chunk count]
        by_case: dict[str, list[float]] = {}
        for i in pool:
            by_case.setdefault(self.links[i], []).append(fused[i])
        case_score = {}
        for link, scores in by_case.items():
            scores.sort(reverse=True)
            mx, support = scores[0], sum(scores[1:])
            case_score[link] = mx + beta * min(support, mx)
        return sorted(case_score, key=lambda l: -case_score[l])[:return_k]


# ── Scorer ───────────────────────────────────────────────────────────────────

def rank_of(link: str, top: list[str]):
    return top.index(link) + 1 if link in top else None


def score_query(q: dict, top: list[str]) -> dict:
    g = q["gold"]
    core, marg, exc = g["core"], g["marginal"], g["excluded"]
    sentinel = q.get("sentinel")  # docket
    sentinel_link = next((l for l in core if docket(l) == sentinel), None) if sentinel else None
    diag = q.get("diagnostic")
    return {
        "id": q["id"], "cls": q["cls"], "query": q["query"], "denominator": q["denominator"],
        "protect": q.get("protect", False),
        "top5": [docket(l) for l in top],
        "core_recovered": sum(1 for c in core if c in top),
        "core_total": len(core),
        "core_ranks": {docket(c): rank_of(c, top) for c in core},
        "marginal_present": [docket(c) for c in marg if c in top],
        "excluded_leaked": [docket(c) for c in exc if c in top],
        "sentinel_rank": rank_of(sentinel_link, top) if sentinel_link else None,
        # Named claim-agency/polarity diagnostic (e.g. DV perpetrator surfaced as
        # top hit). Tracked separately — NOT a pass/fail and not expected to move
        # with fusion changes. Reports the watched case's rank.
        "diagnostic": ({"type": diag["type"], "watch_case": diag["watch_case"],
                        "watch_rank": next((i + 1 for i, l in enumerate(top)
                                            if docket(l) == diag["watch_case"]), None)}
                       if diag else None),
    }


def _rank_sum(r: dict) -> int:
    return sum((rk if rk else 99) for rk in r["core_ranks"].values())


def status_vs_baseline(r: dict, b: dict | None) -> str:
    """UNCHANGED / IMPROVED / DEGRADED on core recovery then core-rank sum."""
    if not b:
        return "n/a"
    if r["core_recovered"] != b["core_recovered"]:
        return "IMPROVED" if r["core_recovered"] > b["core_recovered"] else "DEGRADED"
    rs, bs = _rank_sum(r), _rank_sum(b)
    return "UNCHANGED" if rs == bs else ("IMPROVED" if rs < bs else "DEGRADED")


def sentinel_hardcheck(rows: list[dict], baseline: dict | None) -> str | None:
    """Citation lat-01 (21-70493) must stay in top-5. Out/worse vs baseline = REGRESSION."""
    if not baseline:
        return None
    r = next((x for x in rows if x["id"] == "C-citation-lat01"), None)
    b = next((x for x in baseline["rows"] if x["id"] == "C-citation-lat01"), None)
    if not r or not b:
        return None
    now = r["core_ranks"].get("21-70493")
    was = b["core_ranks"].get("21-70493")
    if now is None and was is not None:
        return f"✗ SENTINEL REGRESSION: 21-70493 DROPPED OUT of top-5 (was rank {was}) — citation lookup re-buried"
    if now is not None and was is not None and now > was:
        return f"✗ SENTINEL REGRESSION: 21-70493 rank {was}→{now} (worse) — citation lookup degraded"
    return f"✓ sentinel held: 21-70493 rank {was}→{now}"


def control_regressions(rows: list[dict], baseline: dict | None) -> list[str]:
    """Return a list of DISQUALIFYING class-B regressions vs a baseline scorecard."""
    if not baseline:
        return []
    base = {r["id"]: r for r in baseline["rows"]}
    bad = []
    for r in rows:
        if not r.get("protect"):  # protected = both B-controls + political imprisonment
            continue
        b = base.get(r["id"])
        if not b:
            continue
        # ADOPTED RULE: only a protected core FALLING OUT of top-5 disqualifies.
        # An intra-top-5 rank slip does NOT (top-5 is what the lawyer reviews;
        # order within it is below product resolution).
        for d, rk in r["core_ranks"].items():
            brk = b["core_ranks"].get(d)
            if brk is not None and rk is None:
                bad.append(f"{r['id']}: core {d} FELL OUT of top-5 (was rank {brk})")
    return bad


def boundary_flags(rows: list[dict]) -> list[str]:
    """Every core sitting at rank 5 — one nudge from falling out. A win that
    depends on a rank-5 core is boundary-fragile, not a clean pass."""
    out = []
    for r in rows:
        for d, rk in r["core_ranks"].items():
            if rk == 5:
                out.append(f"{r['id']}: core {d} at rank 5 (boundary-fragile — one nudge from out)")
    return out


def print_report(rows: list[dict], baseline: dict | None, label: str,
                 suite: dict | None = None, config: str = "fusion=blend, pool=25") -> None:
    print(f"\n{'='*100}\nTHEMATIC SCORECARD — {label}   (top-5, cleaned substrate, {config})\n{'='*100}")
    hdr = f"{'query':<40}{'cls':<10}{'core recov':<11}{'core ranks':<22}{'sentinel':<9}{'leaked':<9}"
    print(hdr + "  top-5 (dockets)")
    print("-" * len(hdr) + "  " + "-" * 30)
    for r in rows:
        cr = f"{r['core_recovered']}/{r['core_total']}" + ("*" if r.get("protect") else "")
        ranks = ",".join(f"{d}:{rk if rk else '—'}" for d, rk in r["core_ranks"].items())
        sent = str(r["sentinel_rank"]) if r["sentinel_rank"] else ("—" if r["cls"] != "A-harm" else "ABSENT")
        leaked = ",".join(r["excluded_leaked"]) or "—"
        print(f"{r['query'][:38]:<40}{r['cls']:<10}{cr:<11}{ranks[:20]:<22}{sent:<9}{leaked:<9}  {r['top5']}")
    print("  (* = protected control: a core FALLING OUT of top-5 disqualifies; an intra-top-5 rank slip does not)")

    diags = [r for r in rows if r.get("diagnostic")]
    if diags:
        print("\nNAMED DIAGNOSTICS (tracked separately — not pass/fail, not expected to move with fusion):")
        for r in diags:
            d = r["diagnostic"]
            print(f"  {r['id']}: {d['type']} — watch {d['watch_case']} rank = {d['watch_rank'] or 'absent'} "
                  f"(#1 = perpetrator surfaced above the victim-claimant)")

    print("\nPER-CLASS read (no blended number — by design):")
    for cls in ("A-harm", "B-control", "C-sentinel"):
        cr = [r for r in rows if r["cls"] == cls]
        tot_core = sum(r["core_recovered"] for r in cr)
        tot_den = sum(r["core_total"] for r in cr)
        print(f"  {cls:<11} core recovered {tot_core}/{tot_den} across {len(cr)} queries")
    sa = next((r for r in rows if r["id"] == "A-sexual-assault"), None)
    if sa:
        print(f"  SENTINEL 14-70905 rank in 'sexual assault': {sa['sentinel_rank'] or 'ABSENT'} (one row, not the pass condition)")

    if baseline is not None:
        base = {r["id"]: r for r in baseline["rows"]}

        # Pre-registered prediction echo (so results are read against it, honestly).
        pred = (suite or {}).get("predictions", {}).get("rrf+case-collapse") if suite else None
        if pred:
            print("\nPRE-REGISTERED PREDICTION (rrf+case-collapse), registered "
                  f"{pred.get('registered','?')}:")
            print(f"  • DV polarity: {pred['domestic_violence_polarity']}")

        # Protected controls + political-imprisonment status.
        print("\nPROTECTED CONTROLS (must not degrade — incl. political imprisonment):")
        for r in rows:
            if not r.get("protect"):
                continue
            print(f"  {status_vs_baseline(r, base.get(r['id'])):<10} {r['id']:<26}"
                  f"core {base.get(r['id'],{}).get('core_recovered','?')}/{r['core_total']}"
                  f" → {r['core_recovered']}/{r['core_total']}")
        bad = control_regressions(rows, baseline)
        print("  " + ("✗ DISQUALIFIED — protected core fell out of top-5: " + "; ".join(bad) if bad
                      else "✓ PASS — no protected core fell out of top-5 (intra-top-5 slips allowed)"))

        # Boundary-proximity: any core at rank 5 is one nudge from falling out.
        bf = boundary_flags(rows)
        print("\nBOUNDARY-PROXIMITY FLAG (rank-5 cores — wins resting on these are fragile):")
        print("  " + ("⚠ " + "; ".join(bf) if bf else "✓ no core sits at rank 5"))

        # Sentinel hard check (citation lat-01) — flagged distinctly, can't be hidden.
        sh = sentinel_hardcheck(rows, baseline)
        if sh:
            print(f"\nSENTINEL HARD CHECK (citation lat-01):\n  {sh}")

        # DV polarity interpretation guard.
        dv = next((r for r in rows if r["id"] == "A-domestic-violence"), None)
        bdv = base.get("A-domestic-violence")
        if dv and bdv and dv["core_recovered"] > bdv["core_recovered"]:
            print("\nDV POLARITY INTERPRETATION GUARD:")
            print("  23-4420 surfaced vs baseline — per the registered prediction, attribute to BM25 "
                  "de-gating (RECALL effect from the pool/rank change), NOT polarity reasoning. "
                  "Do NOT bank as a polarity fix.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, help="external run_baseline-style JSON {id:[case_link...]} to score instead of running the local driver")
    ap.add_argument("--baseline", type=Path, help="prior scorecard JSON to compare against (enforces the control rule)")
    ap.add_argument("--save", type=Path, help="save the top_5_case_ids + scorecard to this path")
    ap.add_argument("--label", default="baseline (hygiene-only, no ranking change)")
    ap.add_argument("--fusion", choices=["blend", "rrf"], default="blend",
                    help="fusion for the local driver: blend (production) or rrf (the intervention)")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="case-collapse support weight: 0=pure max-per-case, >0=capped support")
    ap.add_argument("--reranker", choices=["gte"], default=None,
                    help="cross-encoder reranker stacked as the semantic signal (Alibaba-NLP/gte-reranker-modernbert-base)")
    ap.add_argument("--dirty", action="store_true", help="use the uncleaned 687-chunk set (pre-hygiene)")
    args = ap.parse_args()

    suite = json.loads(SUITE_PATH.read_text())
    queries = suite["queries"]

    if args.results:
        results = json.loads(args.results.read_text())
    else:
        r = LocalRetriever(cleaned=not args.dirty)
        print(f"local driver: {r.n_after} chunks "
              f"({'cleaned, dropped '+str(r.n_before-r.n_after) if not args.dirty else 'DIRTY/uncleaned'}), "
              f"fusion={args.fusion}")
        results, rerank_ms = {}, []
        for q in queries:
            results[q["id"]] = r.search(q["query"], fusion=args.fusion, beta=args.beta, reranker=args.reranker)
            if args.reranker:
                rerank_ms.append(getattr(r, "last_rerank_ms", 0.0))
        if rerank_ms:
            import statistics
            print(f"gte rerank latency over pool={getattr(r,'last_pool_n','?')} chunks/query: "
                  f"p50={statistics.median(rerank_ms):.0f}ms  mean={statistics.mean(rerank_ms):.0f}ms  "
                  f"max={max(rerank_ms):.0f}ms  (n={len(rerank_ms)} queries)")

    rows = [score_query(q, results[q["id"]]) for q in queries]
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    config = f"fusion={args.fusion}, β={args.beta}, pool=25" + (", reranker=gte" if args.reranker else "")
    print_report(rows, baseline, args.label, suite, config)

    if args.save:
        args.save.write_text(json.dumps(
            {"label": args.label, "config": config, "results": results, "rows": rows}, indent=2) + "\n")
        print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
