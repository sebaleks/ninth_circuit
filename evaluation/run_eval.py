"""Evaluate the RAG system on the eval question set.

Metrics (rubric-required):
  - Groundedness:    LLM-as-judge — does the answer's content come from the citations?
  - Citation Accuracy: For each citation, does the snippet support the claim that cites it?
  - Latency p50/p95: from the /chat response's latency_ms field
  - Refusal correctness: out-of-corpus questions should be refused

Usage:
  python evaluation/run_eval.py --against http://localhost:8000
  python evaluation/run_eval.py --against https://rag-api-xxx.onrender.com

Outputs:
  evaluation/results/<date>.json
  evaluation/results/latest.json
  evaluation/results/latest.png  (matplotlib chart)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

import matplotlib.pyplot as plt  # type: ignore
import requests
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Match the user's preferred matplotlib style from feedback_chart_style.md
plt.rcParams.update({
    "figure.facecolor": "#f7f7f7",
    "axes.facecolor":   "#f7f7f7",
    "axes.edgecolor":   "#373737",
    "axes.labelcolor":  "#373737",
    "xtick.color":      "#373737",
    "ytick.color":      "#373737",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "axes.grid": False,
    "font.family": "DejaVu Sans",
})

JUDGE_MODEL = "meta/llama-3.3-70b-instruct"
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "eval_questions.json"
RESULTS_DIR = EVAL_DIR / "results"


def _judge_client() -> OpenAI:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY env var required for LLM-as-judge.")
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=key)


def _judge_call(prompt: str, max_tokens: int) -> str:
    """Call the judge LLM with exponential backoff on 429s."""
    import openai as _openai
    delays = [2, 5, 15, 30, 60]
    for attempt, delay in enumerate(delays + [None]):
        try:
            resp = _judge_client().chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except _openai.RateLimitError:
            if delay is None:
                raise
            print(f"    rate-limited, sleeping {delay}s (attempt {attempt + 1})…")
            time.sleep(delay)
    return ""  # unreachable


def judge_groundedness(answer: str, snippets: list[str]) -> bool:
    """Ask the judge: is EVERY claim in the answer supported by the snippets? YES/NO."""
    if not answer.strip():
        return False
    if not snippets:
        return False
    ctx = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))
    prompt = (
        "You are a strict fact-checker. Given an ANSWER and the SOURCE PASSAGES it was "
        "supposedly drawn from, decide if EVERY factual claim in the ANSWER is supported "
        "by the SOURCE PASSAGES. Respond with exactly YES or NO, then a brief reason.\n\n"
        f"SOURCE PASSAGES:\n{ctx}\n\nANSWER:\n{answer}\n\nResponse:"
    )
    return _judge_call(prompt, max_tokens=120).upper().startswith("YES")


def judge_citation_supports(answer: str, snippet: str) -> bool:
    """Does this one snippet support the claims in the answer it's cited for? YES/NO."""
    prompt = (
        "Does the SNIPPET below substantively support the ANSWER? Reply with exactly YES or NO.\n\n"
        f"SNIPPET:\n{snippet}\n\nANSWER:\n{answer}\n\nReply:"
    )
    return _judge_call(prompt, max_tokens=10).upper().startswith("YES")


def call_chat(base_url: str, question: str, k: int = 5, timeout: int = 90) -> dict:
    resp = requests.post(
        f"{base_url}/chat",
        json={"question": question, "k": k},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def warm_service(base_url: str) -> None:
    """Wake up a cold Render container before we start timing."""
    print(f"Warming {base_url}/health …", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{base_url}/health", timeout=60)
        elapsed = time.perf_counter() - t0
        print(f"{r.status_code} in {elapsed:.1f}s")
    except Exception as e:
        print(f"⚠️  health check failed: {e}")


def run(base_url: str) -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text())["questions"]
    print(f"Loaded {len(questions)} eval questions.\n")

    warm_service(base_url)

    per_question: list[dict] = []
    latencies: list[int] = []

    for i, q in enumerate(questions):
        if i > 0:
            time.sleep(4)  # be polite to NVIDIA free-tier rate limit
        print(f"[{q['id']}] {q['question']}")
        t0 = time.perf_counter()
        try:
            resp = call_chat(base_url, q["question"], k=5)
        except Exception as e:
            print(f"  ⚠️  API error: {e}")
            per_question.append({**q, "error": str(e)})
            continue
        wall_ms = int((time.perf_counter() - t0) * 1000)

        answer = resp.get("answer", "")
        refused = resp.get("refused", False)
        citations = resp.get("citations", [])
        snippets = [c["snippet"] for c in citations]

        # ── Refusal correctness ─────────────────────────────────────────────
        refusal_ok = refused == q["expect_refuse"]

        # ── Groundedness + citation accuracy (skip if refused) ──────────────
        if refused:
            groundedness = None
            cite_correct = None
        else:
            groundedness = judge_groundedness(answer, snippets)
            if citations:
                results = [judge_citation_supports(answer, s) for s in snippets]
                cite_correct = sum(results) / len(results)
            else:
                cite_correct = 0.0  # answer had no citations — automatic 0

        latencies.append(resp.get("latency_ms", wall_ms))

        per_question.append({
            **q,
            "answer": answer,
            "refused": refused,
            "refusal_ok": refusal_ok,
            "groundedness": groundedness,
            "citation_accuracy": cite_correct,
            "n_citations": len(citations),
            "latency_ms": resp.get("latency_ms", wall_ms),
            "wall_ms": wall_ms,
        })

        flag = "✓" if refusal_ok else "✗"
        g_str = "—" if groundedness is None else ("Y" if groundedness else "N")
        c_str = "—" if cite_correct is None else f"{cite_correct:.2f}"
        print(f"  refused={refused} ok={flag}  groundedness={g_str}  cite_acc={c_str}  "
              f"latency={resp.get('latency_ms')}ms")

    # ── Aggregate ────────────────────────────────────────────────────────────
    grounded = [r for r in per_question if r.get("groundedness") is not None]
    cited = [r for r in per_question if r.get("citation_accuracy") is not None]
    refusal = per_question

    summary = {
        "n_questions":       len(per_question),
        "groundedness_pct":  (sum(1 for r in grounded if r["groundedness"]) / len(grounded)) if grounded else 0.0,
        "citation_acc_avg":  (sum(r["citation_accuracy"] for r in cited) / len(cited)) if cited else 0.0,
        "refusal_acc_pct":   sum(1 for r in refusal if r.get("refusal_ok")) / len(refusal),
        "latency_p50_ms":    int(statistics.median(latencies)) if latencies else 0,
        "latency_p95_ms":    int(statistics.quantiles(latencies, n=20)[-1]) if len(latencies) >= 5 else 0,
        "latency_mean_ms":   int(statistics.mean(latencies)) if latencies else 0,
        "n_in_corpus_eval":  len(grounded),
        "n_out_of_corpus":   sum(1 for q in questions if q["expect_refuse"]),
        "evaluated_at_utc":  datetime.now(timezone.utc).isoformat(),
        "rag_url":           base_url,
    }

    result = {"summary": summary, "questions": per_question}
    return result


def write_chart(summary: dict, path: Path) -> None:
    """Three-panel bar chart of the headline metrics, styled per feedback_chart_style.md."""
    BAR = "#30a2da"
    MEAN = "#a50026"
    LABEL = "#373737"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: information quality (groundedness + citation accuracy)
    ax = axes[0]
    vals = [summary["groundedness_pct"] * 100, summary["citation_acc_avg"] * 100]
    labels = ["Groundedness", "Citation Acc."]
    bars = ax.bar(labels, vals, color=BAR, width=0.5)
    ax.axhline(85, color=MEAN, linestyle="--", linewidth=1, label="target 85%")
    ax.set_ylim(0, 105)
    ax.set_ylabel("%", color=LABEL)
    ax.set_title("Information quality")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", color=LABEL, fontsize=10)
    ax.tick_params(left=False, bottom=False)
    ax.legend(frameon=False, fontsize=8)

    # Panel 2: latency
    ax = axes[1]
    p50 = summary["latency_p50_ms"]
    p95 = summary["latency_p95_ms"]
    vals = [p50, p95]
    labels = ["p50", "p95"]
    bars = ax.bar(labels, vals, color=BAR, width=0.5)
    ax.set_ylabel("ms", color=LABEL)
    ax.set_title("Latency")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v}", ha="center", color=LABEL, fontsize=10)
    ax.tick_params(left=False, bottom=False)

    # Panel 3: refusal accuracy
    ax = axes[2]
    vals = [summary["refusal_acc_pct"] * 100]
    bars = ax.bar(["Refusal correctness"], vals, color=BAR, width=0.4)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%", color=LABEL)
    ax.set_title("Guardrails")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", color=LABEL, fontsize=10)
    ax.tick_params(left=False, bottom=False)

    plt.tight_layout()
    plt.savefig(path, dpi=110, facecolor="#f7f7f7")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--against", default=os.environ.get("RAG_API_URL", "http://localhost:8000"),
                        help="Base URL of the RAG API (default: RAG_API_URL env or http://localhost:8000)")
    args = parser.parse_args()
    base = args.against.rstrip("/")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = run(base)

    s = result["summary"]
    print("\n=== SUMMARY ===")
    print(f"  Groundedness:        {s['groundedness_pct'] * 100:.1f}% over {s['n_in_corpus_eval']} in-corpus answers")
    print(f"  Citation accuracy:   {s['citation_acc_avg'] * 100:.1f}% (avg per-citation support)")
    print(f"  Refusal correctness: {s['refusal_acc_pct'] * 100:.1f}%")
    print(f"  Latency p50/p95:     {s['latency_p50_ms']} / {s['latency_p95_ms']} ms")

    # Write outputs
    date_stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dated_path = RESULTS_DIR / f"{date_stamp}.json"
    dated_path.write_text(json.dumps(result, indent=2))
    (RESULTS_DIR / "latest.json").write_text(json.dumps(result, indent=2))
    write_chart(s, RESULTS_DIR / "latest.png")

    print("\nWrote:")
    print(f"  {dated_path}")
    print(f"  {RESULTS_DIR / 'latest.json'}")
    print(f"  {RESULTS_DIR / 'latest.png'}")


if __name__ == "__main__":
    main()
