"""Unit tests for pipeline/rag_ingest.py.

Covers the pure helpers (chunking, normalization, nlist computation). The full
ingest run is network-dependent (NVIDIA + PDF downloads) and is exercised by
the manual `python pipeline/rag_ingest.py` rather than CI.
"""

from __future__ import annotations

import numpy as np
import pytest
import tiktoken

from pipeline.rag_ingest import (
    EMBED_DIM,
    chunk_page,
    compute_nlist,
    l2_normalize,
)


# ── compute_nlist auto-scaling ───────────────────────────────────────────────

@pytest.mark.parametrize("n,expected_floor", [
    (10,    4),    # tiny corpus floors at 4
    (180,   4),    # 30-case MVP: ~6 cells (capped by N/30)
    (3000,  50),   # 500-case: ~100 cells
    (36000, 700),  # full corpus: ~759 cells
])
def test_compute_nlist_scales(n, expected_floor):
    nlist = compute_nlist(n)
    assert nlist >= 4
    assert nlist >= expected_floor or n // 30 < expected_floor


def test_compute_nlist_always_min_4():
    assert compute_nlist(0) == 4
    assert compute_nlist(1) == 4


# ── L2 normalization ─────────────────────────────────────────────────────────

def test_l2_normalize_unit_length():
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((5, EMBED_DIM)).astype(np.float32)
    normed = l2_normalize(vecs)
    norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_l2_normalize_handles_zero():
    vecs = np.zeros((1, EMBED_DIM), dtype=np.float32)
    normed = l2_normalize(vecs)
    assert np.all(np.isfinite(normed))    # no NaNs from divide-by-zero


def test_l2_normalize_dtype():
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((3, EMBED_DIM)).astype(np.float32)
    assert l2_normalize(vecs).dtype == np.float32


# ── chunk_page ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def encoder():
    return tiktoken.get_encoding("cl100k_base")


def test_chunk_page_short_text_one_chunk(encoder):
    text = "This is a short page that fits in one chunk."
    chunks = chunk_page(text, encoder, tokens_per_chunk=1500, overlap=150)
    assert len(chunks) == 1
    chunk_text, char_start, char_end, n_tokens = chunks[0]
    assert chunk_text == text
    assert char_start == 0
    assert char_end == len(text)
    assert n_tokens > 0


def test_chunk_page_empty():
    enc = tiktoken.get_encoding("cl100k_base")
    assert chunk_page("", enc) == []
    assert chunk_page("   ", enc) == []


def test_chunk_page_splits_long_text(encoder):
    # Generate text longer than 200 tokens
    text = ("The court considered the petition for review. " * 100).strip()
    chunks = chunk_page(text, encoder, tokens_per_chunk=100, overlap=20)
    assert len(chunks) >= 2
    # Each chunk respects the token budget
    for chunk_text, _, _, n_tokens in chunks:
        assert n_tokens <= 100


def test_chunk_page_overlap_creates_continuity(encoder):
    text = ("alpha beta gamma " * 200).strip()
    chunks = chunk_page(text, encoder, tokens_per_chunk=60, overlap=15)
    # Adjacent chunks should share some token content (overlap)
    assert len(chunks) >= 2
    assert chunks[0][3] == 60   # first chunk uses full window
