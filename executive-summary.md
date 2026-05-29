# Executive Summary — Ninth Circuit Asylum RAG System

A Retrieval-Augmented Generation (RAG) search system over U.S. Ninth Circuit asylum
opinions, integrated into the existing `asylum-viewer` Next.js app as a "Case Search"
panel on the right side of `/cases`. Built entirely on free-tier infrastructure.

## System

**Architecture (3 services + existing Supabase):**

- **Frontend** — Next.js on Vercel. A search panel calling `/api/rag/*` proxy routes that
  forward server-side to `$RAG_API_URL` (the backend URL is never exposed to the browser).
- **Backend** — FastAPI on Render free tier (512 MB RAM). Endpoints: `/chat`, `/search`,
  `/health`.
- **Vectors** — FAISS `IndexIVFPQ` + Parquet metadata, versioned in Git LFS (~3 MB).
- **Inference** — NVIDIA NIM free tier: embed `nvidia/llama-nemotron-embed-1b-v2`
  (2048-dim), rerank `nvidia/llama-nemotron-rerank-1b-v2`, generate
  `meta/llama-3.3-70b-instruct`.

**Retrieval pipeline:** FAISS dense top-N → NVIDIA rerank → BM25 hybrid
(`0.6 * rerank + 0.4 * bm25`) → dedupe by case → top-k. Out-of-corpus queries are refused
via a dense-cosine threshold (0.15), which is more uniformly calibrated than the rerank
sigmoid.

**Footprint:** ~170 MB RAM at full load (33% of Render's 512 MB); BM25 adds only +9.5 MB
at 30-case scale. Git LFS usage ~3 MB (0.3% of the 1 GB free tier).

## How it meets the Quantic AI Engineering rubric (score 5)

| Rubric requirement | Where it is met |
|---|---|
| Outstanding RAG, correct responses, matching citations | Dense + NVIDIA rerank + BM25 hybrid in `rag_api/retrieval.py`; citations resolved to chunk IDs with snippet, page, and case link |
| Ingest and indexing works | `pipeline/rag_ingest.py` downloads PDFs, page-aware ~1500-token chunking, builds `data/index.faiss` + `data/metadata.parquet` |
| Excellent, well-structured architecture | Clean 3-tier separation, all free-tier, no external vector DB (FAISS-in-Git-LFS) |
| Public deployment fully functional | Render (backend) + Vercel (frontend), both public, both free |
| CI/CD on push/PR | Three GitHub Actions workflows: `rag-api-test`, `rag-api-deploy`, `rag-eval` |
| Excellent design docs | `design-and-evaluation.md`, `ai-tooling.md`, `deployed.md`, README RAG section |
| Excellent evaluation (groundedness, citation accuracy, latency) | `evaluation/run_eval.py` — LLM-as-judge groundedness, citation-accuracy check, latency p50/p95, 20-question set, matplotlib chart |
| Excellent demo | 8-minute screen-share script documented in the project plan |

## Design decisions worth noting

- **FAISS in Git LFS** instead of a hosted vector DB — keeps the system to three services
  with zero recurring cost, and the index is a reproducible, versioned artifact.
- **BM25 hybrid** added on top of dense+rerank so literal legal terms (e.g. a country name
  like "Honduras") surface the exact-match passages instead of only semantic neighbors.
- **Dedupe by case** so each unique case appears once in results — lawyers want a distinct
  list of cases, not repeated chunks of the same opinion.
- **Search-only UX** — the panel returns ranked cases with snippets and citations rather
  than a generated essay, matching how attorneys actually consume results.
