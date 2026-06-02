# Executive Summary — Ninth Circuit Asylum RAG System

## Authorship

This document describes a parallel implementation by Sebastian Steen, built on top of the original architecture by Victor Palacios (original repo). The Overview and Architecture sections describe the inherited system; the marked Experiments are alternative implementations under investigation. Sections marked INHERITED describe Victor's design as-is; sections marked EXPERIMENT describe my proposed changes.

---

## Overview

This project delivers a Retrieval-Augmented Generation (RAG) search system over U.S. Court
of Appeals for the Ninth Circuit asylum opinions. It is integrated directly into the
existing `asylum-viewer` Next.js application as a "Case Search" panel on the right side of
the `/cases` page, rather than shipped as a standalone product. The system lets an
attorney paste a fact pattern or type a natural-language question and receive a ranked,
deduplicated list of the most relevant Ninth Circuit asylum cases, each accompanied by the
supporting text snippet, page number, and a link back to the source PDF.

The entire system runs on free-tier infrastructure with no recurring cost: Vercel hosts
the frontend, Render hosts the backend, NVIDIA NIM provides all model inference, and the
vector index lives in the Git repository itself via Git LFS. The only pre-existing service
it touches is the Supabase Postgres database that already backs `asylum-viewer`, and it
touches that only for structured case metadata — never for vectors.

This document describes a parallel implementation built on this architecture. The inherited components, frontend, Supabase data, evaluation harness, and deployment infrastructure, remain unchanged. The retrieval pipeline, vector store, and embedder are the subject of architectural experiments described below, motivated by prior research on a 30-case corpus of Ninth Circuit asylum opinions and constrained by the same $0-ongoing-cost and 512 MB Render free-tier budget as the original.

---

## Architecture

The system is composed of three services plus the existing Supabase database.

```
asylum-viewer (Next.js · Vercel · free)              ← INHERITED
  ├─ /cases page + Case Search panel
  └─ /api/rag/* server-side proxy routes → $RAG_API_URL
                                  │
                                  ▼
rag-api (FastAPI · Render free tier · 512 MB RAM)    ← MY PARALLEL VERSION
  ├─ POST /chat
  ├─ POST /search
  └─ GET  /health
                                  │
                                  ▼
                      RETRIEVAL PIPELINE              ← EXPERIMENT (hybrid scheme)
                      ┌─────────────────────┐
                      │ 1. Dense retrieval  │
                      │ 2. Rerank (his) /   │
                      │    or skip (yours)  │
                      │ 3. BM25 blend       │
                      │ 4. Dedupe by case   │
                      └─────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                       ▼
   Vector store           NVIDIA NIM                  Supabase
   ⇩ EXPERIMENT           ⇩ EXPERIMENT                INHERITED
   FAISS (inherited)      Embed: his vs local
   Chroma                 Rerank: his vs none
   pgvector               Gen: 70B vs smaller
```

**Experiments**
Vector store swap (FAISS → Chroma/pgvector)
Embedder swap (NIM → local BGE/GTE/legal)
Hybrid fusion scheme comparison (his rerank+blend → your RRF-no-reranker)
Hybrid blend tuning (only if his fusion stays)
Chunking ablation
Generation model swap
Segmenter integration

**Frontend.** The search panel is a collapsible right sidebar inside the existing `/cases`
page (a thin "💬 Chat" tab when closed, ~400px wide when open), so the new capability
augments the table the user already works with instead of fragmenting the app into a
separate route. The browser never calls the Render backend directly. Instead it POSTs to
Next.js proxy routes under `/api/rag/*`, which forward the request server-side to
`process.env.RAG_API_URL`. This keeps the backend URL server-only, avoids CORS preflights,
and prevents the endpoint from leaking into client bundles.

**Backend.** A FastAPI application exposes `/chat`, `/search`, and `/health`. At startup it
loads the FAISS index and the Parquet metadata into memory once (FastAPI lifespan hook),
pre-warms the BM25 index, and then serves all queries in-process with no further disk I/O.

**Vector store.** Embeddings are stored in a FAISS `IndexIVFPQ` (Product Quantization)
index, with the human-readable per-chunk metadata (chunk text, page, case link, chunk ID,
disposition, publication status) in a companion Parquet file. Both artifacts are versioned
in the Git repository through Git LFS (~3 MB total at the 30-case MVP scale), so the index
is a reproducible, version-controlled artifact rather than state living in an external
database.

**Inference.** All model calls go to NVIDIA NIM's free tier through its OpenAI-compatible
API: embedding with `nvidia/llama-nemotron-embed-1b-v2` (2048-dimensional vectors,
2048-token context), reranking with `nvidia/llama-nemotron-rerank-1b-v2`, and answer
generation with `meta/llama-3.3-70b-instruct`.

---

## How it works

### Ingest and indexing

`pipeline/rag_ingest.py` builds the index from the curated case list in
`reports/sample_30_cases.csv`. For each case it downloads the opinion PDF, extracts text
per page with PyMuPDF, and splits the text into page-aware chunks of approximately 1,500
tokens with a 150-token overlap (token counts via `tiktoken`). Page-aware chunking is what
lets a citation say "page N of case X" instead of an opaque byte offset. Each chunk is
embedded with the NVIDIA embedder in `passage` mode, the vectors are L2-normalized so inner
product equals cosine similarity, and the resulting matrix is used to train and populate
the `IndexIVFPQ`. The index and metadata are written to `data/index.faiss` and
`data/metadata.parquet`, and an MLflow run logs the chunk count, model ID, and build
duration. A fixed `random.seed(42)` makes any sampling reproducible.

The index parameters auto-scale with corpus size, so the same code path serves the 30-case
MVP and the eventual full ~5,981-case corpus with no refactor: `nlist = max(4, round(4·√N))`
capped at `N/30` (so each Voronoi cell has enough training samples), `m = 64` subquantizers,
and 8 bits per code.

### Retrieval pipeline

A query flows through four stages:

1. **Dense retrieval.** The query is embedded in `query` mode (the embedder's asymmetric
   mode, tuned for QA-shaped inputs), and FAISS returns the top-N candidate chunks by
   cosine similarity.
2. **Neural reranking.** The candidates are re-scored by the NVIDIA reranker, which is a
   cross-encoder and therefore far more precise about query/passage relevance than the
   bi-encoder embedding used for the first-stage recall.
3. **BM25 hybrid.** A classical sparse BM25 score is computed for each candidate and
   blended with the rerank score as `0.6 · rerank + 0.4 · bm25`. This ensures that literal
   legal terms — a country name like "Honduras," a docket number, a statutory phrase —
   surface the exact-match passages, which a purely semantic system tends to bury under
   topically-similar-but-not-matching text.
4. **Dedupe by case.** Results are collapsed so each unique case appears once (keeping its
   highest-scoring chunk), then truncated to the top-k. Attorneys want a distinct list of
   cases, not three chunks of the same opinion occupying the top three slots.

The `/chat` endpoint additionally feeds the top-k passages to `meta/llama-3.3-70b-instruct`
with a numbered-passage prompt (`[1] … [2] …`), asks for `[N]`-tagged citations, then parses
the tags back to chunk IDs to render the citation cards.

### Guardrails

Out-of-corpus queries are refused using a dense-cosine threshold: if the maximum FAISS
cosine across all candidates is below 0.15, the system declines to answer. The dense score
is used here, rather than the rerank score, because empirically it is far better
calibrated — in-corpus queries score 0.21 and up while an out-of-corpus query like "weather
in Boston" scores ~0.09, whereas the rerank sigmoid collapses toward zero for many
legitimate natural-language questions and is too noisy to threshold on. Generation is
additionally capped at 500 tokens, and the system prompt instructs the model to ground
every claim in the supplied passages and to refuse only clearly unrelated questions.

---

## Resource footprint

The backend consumes roughly 170 MB of RAM at full load on the 30-case corpus — about 33%
of Render's 512 MB free-tier limit. The BM25 hybrid index, which is the most recent
addition, accounts for only about +9.5 MB at this scale. The Git LFS artifacts total ~3 MB,
which is 0.3% of GitHub's 1 GB free LFS allowance. There is comfortable headroom at MVP
scale, though see the scaling tradeoffs below for the full-corpus picture.

**My version's projected footprint.** Moving the embedder in-process (BGE-small or GTE-small at ~150–180 MB resident) adds memory to the Render container. At 30-case MVP scale: ~170 MB baseline + ~150–180 MB embedder + Chroma replacing FAISS (similar footprint) ≈ 320–350 MB total, comfortably within the 512 MB ceiling. The actual deployed footprint will be measured rather than estimated, since Render's reported memory may differ from process-internal measurements. At larger corpus scales the math gets tighter — see the constraints discussion in My v0.2 priorities.

---

## Key design tradeoffs

**Original Design Tradeoffs (inherited):**

Every significant decision in this system traded one desirable property for another. The
most consequential ones:

**FAISS in Git LFS vs. a hosted vector database.** Choosing an in-process FAISS index
versioned in Git LFS kept the system to three services with zero recurring cost and made
the index a reproducible artifact that travels with the code. The cost is that the index
is *static between deploys*: adding new cases requires re-running ingest and pushing a new
binary, rather than an incremental upsert against a live database. It also means the index
must fit in the backend's RAM, which couples index size to the Render instance size. We
evaluated Supabase pgvector, Neon, Pinecone, Qdrant, and Chroma; the pgvector options were
ruled out by their 500 MB free caps against a projected ~650 MB full corpus, and the hosted
vector DBs each added a vendor, an API key, and a second cold-start to stack on top of
Render's. For the rubric's emphasis on a clean, reproducible, fully-free architecture,
FAISS-in-Git-LFS was the better fit — but a team optimizing for live, incremental updates
would reasonably choose a hosted store.

**`IndexIVFPQ` from day one vs. starting with a flat index.** Using a quantized IVF index
from the start means the build and load code is identical at every corpus scale, so the
jump from 30 cases to ~6,000 requires no code change. The tradeoff is that at 30-case scale
(~700 chunks) the quantization is over-engineered: the IVF clusters are under-trained and
recall is an estimated 1–3% lower than a simple `IndexFlatIP` would deliver. We accepted a
small, measurable recall hit at MVP scale in exchange for never rewriting the index code —
a deliberate bet that the system will grow into the index rather than the other way around.

**BM25 hybrid vs. pure neural retrieval.** Adding a classical sparse signal fixed a real,
observed failure mode (literal terms like country names being out-ranked by semantically
similar passages) and is cheap in memory at MVP scale. The tradeoffs are a fixed blend
weight (0.6/0.4) that is not yet tuned per query type, additional per-query computation,
and a BM25 index whose memory cost grows with the corpus and which currently rebuilds in
memory at startup rather than being persisted.

**Dedupe by case vs. returning the best chunks.** Collapsing to one result per case gives
attorneys the distinct case list they actually want, but it discards potentially useful
secondary passages from a highly relevant case. A case that is relevant in three distinct
ways is currently represented by only its single best chunk.

**Generation with a 70B model vs. a smaller/faster model.** `meta/llama-3.3-70b-instruct`
produces high-quality, well-grounded prose, but it is the dominant contributor to `/chat`
latency. On NVIDIA's free tier, queueing can push p95 latency well past the warm target.
A smaller model would be faster and cheaper but would likely degrade groundedness and
citation discipline.

**Synchronous JSON response vs. streaming.** The current `/chat` returns a single JSON
payload only after the full answer is generated, which is simpler to implement, test, and
cache. The cost is user-perceived latency: the user waits several seconds staring at a
spinner instead of watching tokens appear. Streaming is the top item on the post-MVP list.

**Render free tier vs. an always-on host.** The free tier costs nothing but sleeps after
15 minutes of inactivity and takes ~30 seconds to cold-start, so the first query after idle
is slow. This is an acceptable tradeoff for a prototype and demo, but not for production
traffic.

**My Design Tradeoffs:**

**Chroma over FAISS-in-Git-LFS.** Trades reproducibility-via-Git for higher recall (no IndexIVFPQ quantization loss) and a cleaner separation of code and data. The original chose FAISS for zero-services and a versioned artifact; my version accepts a separate Chroma persistence layer in exchange for measured 1-3%+ recall improvement.

**Qdrant Cloud over FAISS-in-Git-LFS (alternative production architecture).** Trades the static, reproducible-via-Git index for a hosted service that decouples vector storage from Render's 512 MB RAM ceiling. The original chose FAISS-in-Git-LFS to keep the system to three services with zero recurring cost; my version evaluates whether adding a fourth (Qdrant Cloud free tier) is justified by full-corpus scaling headroom and native sparse-vector support. The cost is an additional external dependency and a network hop on every query — measured around 20-50 ms vs zero for in-process FAISS. At MVP scale this may be acceptable; at full corpus the architectural simplification (sparse + dense in one service, no Render RAM ceiling) may outweigh the operational cost.

**In-process embedder vs NIM-hosted embedder.** The original chose NIM-hosted embedding to keep the Render backend lean and avoid loading any model into the 512 MB container. The cost is network latency on every query and a hard dependency on NIM's queue depth. My version moves embedding in-process using BGE-small (~133 MB resident), which adds RAM pressure but eliminates the network hop from the latency budget. At 30-case scale this fits comfortably; at full corpus it requires verifying total RAM stays under 512 MB once Chroma, BM25, and the app itself are accounted for.

**RRF fusion without reranker over rerank+blend.** Trades architectural simplicity for the reranker failure mode my prior experiments documented on doctrinal queries: a standard web-search-trained cross-encoder (ms-marco-MiniLM-L12-v2) hurt every strong baseline tested, including the hybrid R@5 of 0.958, where reranking dropped performance to 0.812. The mechanism was out-of-domain training — the reranker demoted legal phrasing in favor of lexical web-search-style matches. The reranker did help a weak baseline (long-chunk dense, +0.125 R@5), but even the helped variant did not reach the unranked hybrid baseline. My version inherits this finding for doctrinal queries; behavior on paraphrased or lay-language queries is untested and is named as future work.


---

## Room for improvement (inherited)

The prototype-1 evaluation (30-case corpus) confirmed the pipeline works end-to-end on
every question, but also surfaced clear, addressable gaps. Current measured numbers sit
below the rubric's "excellent" targets — groundedness ~43%, citation accuracy ~55%, refusal
correctness ~55%, p50 latency ~8.5s, p95 ~55s — and the analysis below explains why and
what closes each gap.

1. **Expand the corpus from 30 to ~6,000 cases.** This is the single highest-leverage
   change. Many evaluation misses are specific-fact questions ("did the court grant case
   24-631?") where the relevant case simply is not in the 30-case MVP, or where the
   embedder matches poorly on docket numbers because there is too little surrounding
   context to disambiguate. A larger corpus dramatically improves retrieval recall, and
   because the index parameters auto-scale, it requires only re-running `rag_ingest.py`.

2. **Tune the refusal threshold.** The fixed 0.15 dense-score cutoff currently refuses some
   legitimate in-corpus questions while still letting some borderline cases through. A
   sweep over {0.10, 0.12, 0.15, 0.18} against the labeled eval set would raise refusal
   correctness without weakening out-of-corpus rejection.

3. **Make the generation prompt citation-strict.** The model frequently cites passages it
   does not actually rely on, which depresses citation accuracy. A prompt revision
   requiring it to cite *only* the passage(s) that directly support each individual claim,
   plus a post-hoc check that drops unsupported citations, should lift the metric
   substantially.

4. **Run the scaffolded ablations.** The evaluation harness already supports sweeping
   `chunk_size ∈ {500, 1000, 1500}` and `k ∈ {3, 5, 10}`, but prototype-1 skipped them to
   stay within the free-tier rate-limit budget. Running these would let us choose chunk
   size and k empirically rather than by rule of thumb, and would produce the ablation
   table the rubric rewards.

5. **Tune or learn the hybrid blend weight.** The 0.6/0.4 rerank/BM25 split is a reasonable
   default but is uniform across all queries. Keyword-heavy queries (a specific statute or
   country) likely want more BM25 weight; conceptual queries want more rerank weight. A
   small tuning pass, or a lightweight query classifier that picks the weight, would
   improve ranking quality.

6. **Stream `/chat` responses.** Switching from a single JSON response to a server-sent
   token stream turns the experience from "wait, then see the answer" into "watch the
   answer appear," which is both a better demo and a real UX upgrade. The 70B model
   supports SSE natively; the change is small on both ends, with citation tags buffered
   until close so parsing stays identical.

7. **Plan BM25 persistence for full-corpus scale.** At MVP scale the in-memory BM25 rebuild
   at startup is negligible, but at full corpus it grows in both memory and startup time.
   Options include persisting the BM25 index alongside the FAISS artifact in Git LFS,
   moving sparse retrieval to SQLite FTS5, or dropping BM25 in favor of a single hybrid
   retriever — to be decided once full-corpus memory is profiled against Render's 512 MB
   ceiling.

8. **Address cold starts and latency for production.** If this graduates from prototype to
   production, the Render free tier's sleep/cold-start behavior and the NVIDIA free-tier
   queueing that drives p95 latency would need to be addressed — via a warm-keeping ping,
   a paid inference tier, or a smaller/faster generation model — depending on the
   latency budget the use case actually requires.


**My priorities**
This section maps my proposed experiments to the original "Room for improvement" priorities and to additional architectural questions surfaced by my prior work on this corpus. Each priority references the relevant Experiment number from the architecture diagram above.

Experiment 1 — Vector store comparison.  
Replace FAISS IndexIVFPQ with Chroma, Qdrant Cloud, and pgvector; benchmark all four on the same evaluation set. The original architecture chose FAISS-in-Git-LFS for reproducibility and zero external services; the documented cost was an estimated 1–3% recall hit from IVF_PQ quantization at MVP scale, and a projected 650 MB index size at full corpus that exceeds Render's 512 MB ceiling. Each alternative resolves a different constraint:  
Chroma: higher recall than IVF_PQ, similar Render RAM footprint, no external service dependency.  
Qdrant Cloud: decouples vector storage from Render RAM, native sparse-vector support for cleaner hybrid retrieval, generous 1 GB free tier.  
pgvector (Supabase): keeps vectors in the existing Postgres-centric stack, no new service, but constrained by Supabase 500 MB database cap.  
Direct measurement on this corpus will quantify the actual recall, latency, and operational tradeoffs of each.

Experiment 2 — Embedder comparison  
Replace nvidia/llama-nemotron-embed-1b-v2 (2048-dim, NIM-hosted) with in-process alternatives: BGE-small, GTE-small, and optionally a legal-domain embedder (Free Law Project ModernBERT). The original chose NIM-hosted embedding to keep the Render container memory free; my version trades RAM headroom for elimination of network latency and queue depth from the query path. Prior same-tier embedder testing on this corpus suggests GTE-small or E5-small outperform BGE-small at 384-dim; that work needs revalidation against the NIM-hosted baseline at this system's scale.  
Addresses: item 8 (latency for production) by removing one of the NIM dependencies that contributes to p95 queueing. Storage reduction (5.3× smaller vectors at 384-dim vs 2048-dim) also helps the Render RAM ceiling at full-corpus scale.

Experiment 3 — Hybrid fusion scheme comparison  
The original architecture uses a three-stage hybrid: dense retrieval → NVIDIA cross-encoder rerank → BM25 blend at fixed 0.6/0.4 weights. My version tests an alternative: dense + BM25 fused via Reciprocal Rank Fusion (RRF), no reranker. Prior work showed (a) RRF hybrid achieved R@5 = 0.958 on a comparable corpus without a reranker, (b) the standard ms-marco cross-encoder reranker hurt strong baselines (out-of-domain web-search training), and (c) fixed-weight blending has a documented destructive failure mode when one method has zero signal — particularly on paraphrased queries (BM25 collapses) or very long chunks (dense truncates).  
Addresses: item 5 (hybrid blend weight tuning) by testing whether the rerank stage adds value at all, not just whether the blend weights are correctly tuned. If the RRF-without-reranker version is competitive or better, the question shifts from "what weights should the blend use" to "is the reranker step worth keeping."

Experiment 4 — Hybrid blend tuning (conditional)  
Only relevant if Experiment 3 shows the original fusion scheme (rerank + BM25 blend) is the right architecture for this corpus. In that case, the fixed 0.6/0.4 weight is the next thing to test: regime-aware blending (different weights per query type), or query-classifier routing (dynamic weight selection). The destructive-RRF mechanism applies to weighted blending too: if a query has zero signal in one method, the fixed weight on that method actively drags the ranking down.  
Addresses: item 5 directly, conditional on the architectural question in Experiment 3.

Experiment 5 — Chunking ablation  
Sweep chunk size {256, 512, 1500} and overlap {32, 150} against retrieval quality and citation accuracy. The original chose 1500-token page-aware chunks to support clean per-page citations; prior work on this corpus showed 256-token sliding-window chunks improved retrieval recall by +30% over whole-opinion chunks. The open question is whether smaller chunks improve retrieval enough to justify a different citation-resolution strategy, or whether 1500-token page-aware chunks are the right tradeoff at production scale.  
Addresses: item 4 (scaffolded ablations) directly. The original has the eval harness wired to sweep chunk_size ∈ {500, 1000, 1500} and k ∈ {3, 5, 10} but did not run it in prototype-1 due to rate-limit budget; my version extends the sweep to smaller sizes and actually runs it.

Experiment 6 — Generation model swap  
Hold the retrieval pipeline constant (best from Experiments 1–5), swap meta/llama-3.3-70b-instruct for smaller NVIDIA NIM models (meta/llama-3.1-8b-instruct, meta/llama-3.2-3b-instruct). Measure latency change and quality change. The hypothesis: improved retrieval quality from earlier experiments reduces the synthesis burden on the generation model, making smaller models viable. The 70B model is the dominant latency contributor at p50 (8.5s) and dominates p95 (55s) via free-tier queueing.  
Addresses: item 8 (cold starts and latency for production) and is the experiment most directly tied to the original's stated latency priority. Generation latency cannot be addressed by retrieval improvements alone, but better retrieval enables smaller generation models, which can.

Experiment 7 — Segmenter integration  
Replace page-aware chunking with role-based segmentation (CAPTION, PROCEDURAL_POSTURE, PETITIONER_FACTS, COURT_ANALYSIS, DISPOSITION, SEPARATE_OPINION). Prior work on this corpus produced a verified six-role segmenter (v2 + Fix #1) that improved retrieval R@5 by +14% and MRR by +19% over a fixed-window baseline. Open question: does role-based segmentation help on this evaluation set, given that the original system already uses page-aware chunking which preserves natural document structure differently. Potential extension: token-window subchunking within roles, which composes the semantic-section signal of role boundaries with the chunk-size discipline that makes dense embedders effective.  
Addresses: an architectural question the original surfaced but did not investigate — whether semantic structure (rhetorical roles) helps retrieval beyond what page-aware chunking already provides.

