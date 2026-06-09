"""Prompt assembly, LLM call, and citation-tag parsing."""

from __future__ import annotations

import re

from rag_api import nvidia_client, timing


_CITE_RE = re.compile(r"\[(\d+)\]")


def parse_citations(answer: str, n_passages: int) -> list[int]:
    """Extract 1-indexed [N] tags from the answer; return unique 0-indexed passage IDs in
    order of first appearance, filtered to valid range."""
    seen: set[int] = set()
    ordered: list[int] = []
    for match in _CITE_RE.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= n_passages and (n - 1) not in seen:
            seen.add(n - 1)
            ordered.append(n - 1)
    return ordered


def answer_with_citations(question: str, hits: list[dict]) -> tuple[str, list[dict]]:
    """Generate an answer using `hits` as context. Returns (answer_text, used_hits).

    `used_hits` is filtered to only the passages the model actually cited (in citation
    order). If the model cites nothing, returns all hits as fallback evidence.
    """
    passages = [h["snippet"] for h in hits]
    # Timed at the dispatch point; gross time has any retry backoff subtracted
    # back out in build_report via generate_retry_sleep_ms.
    with timing.timer("generate"):
        answer = nvidia_client.generate(question, passages)

    cited_idxs = parse_citations(answer, len(passages))
    used = [hits[i] for i in cited_idxs] if cited_idxs else hits
    return answer, used
