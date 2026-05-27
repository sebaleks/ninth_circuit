# Deployed URLs

Public, free-tier deployment of the Ninth Circuit asylum-case RAG system.

## Frontend (Vercel)

- **URL**: <https://asylum-viewer.vercel.app/cases>
- **What to do**: open `/cases`, click the **💬 Chat** tab on the right edge of the page
  to expand the chat panel.
- **Hosting**: Vercel free tier, automatic on every push to `main`.
- **Source**: `asylum-viewer/` in this repo.

> ⚠️ The chat + similar-case search are in **TESTING MODE**. Answers may be incorrect —
> verify against the cited PDFs. A yellow banner inside the chat panel says the same
> thing.

## Backend (Render)

- **Health URL**: `https://rag-api-<name>.onrender.com/health` (replace `<name>` with the
  actual Render service name from the dashboard once provisioned)
- **Hosting**: Render free tier — 512 MB RAM, sleeps after 15 min idle, ~30 s cold start
- **Source**: `rag_api/` in this repo
- **Endpoints**:
  - `GET  /health` → `{ status, n_chunks, embed_model, rerank_model, gen_model, build_sha }`
  - `POST /search` → `{ hits[] }`
  - `POST /chat`   → `{ answer, citations[], latency_ms, refused }`

## Vector index (Git LFS)

- `data/index.faiss` and `data/metadata.parquet` are versioned in this repo via Git LFS.
- The Render backend pulls them at container build time via `git lfs pull`.

## Inference (NVIDIA NIM)

- Free-tier key via <https://build.nvidia.com>.
- Models in active use (May 2026):
  - Embed:  `nvidia/llama-nemotron-embed-1b-v2`
  - Rerank: `nvidia/llama-nemotron-rerank-1b-v2`
  - Gen:    `meta/llama-3.3-70b-instruct`

## CI/CD

GitHub Actions workflows in `.github/workflows/`:

- `rag-api-test.yml` — runs on PR / push touching `rag_api/**` or `data/**`
- `rag-api-deploy.yml` — on push to `main`, hits Render's deploy hook
- `rag-eval.yml` — `workflow_dispatch` + weekly cron; commits `evaluation/results/<date>.json`

## Grader access

The repo is shared with **`quantic-grader`** on GitHub per the rubric submission
requirement. They can clone, run `git lfs pull`, and follow `README.md` to reproduce
everything locally.
