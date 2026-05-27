# AI tooling — how this project was built with Claude Code

Per the Quantic project rubric, this document describes the AI code tooling used to build
the RAG layer and how it was used.

## Tools used

- **Claude Code (CLI)** by Anthropic — primary coding agent. Multi-turn, file-aware
  agent with shell + edit access. Used for ~95% of the RAG-layer code.
- **Claude Sonnet 4.6** — the underlying model behind Claude Code for most of the
  session. Switched to **Claude Opus 4.7** for the longer planning + research turns.
- **Vercel Build** + **Next.js compiler** — for build-time validation of the chat-panel
  UI (no local dev server used, per a project rule that "preview tools break Claude
  Code"). The agent ran `npm run build` after frontend edits to verify type/compile
  errors before committing.
- **NVIDIA build.nvidia.com** model catalog + raw `curl` probes — used to identify which
  embed/rerank models were still alive after NVIDIA's 2026-05-18 deprecation wave (the
  originally-specified `llama-3.2-nv-{embedqa,rerankqa}-1b-v2` were EOL'd; replaced with
  the `llama-nemotron-*` successors).

## What worked well

- **Planning before coding.** The agent went through four plan revisions
  (`/Users/victor/.claude/plans/ok-i-want-to-rippling-peach.md`) before writing any code
  — switching vector stores from pgvector → Pinecone → FAISS+LFS as I refined
  constraints, removing GCP/Vertex/Cloud Run references at my request, and adjusting
  embedding dim and chunk size based on model specs. Each revision was a single file edit
  I could review in plain text. This caught a lot of "we'd-have-to-rewrite-it-later"
  decisions cheaply.
- **Tool-use parallelism.** When exploring the codebase the agent ran multiple Bash /
  Read calls in parallel, which made the initial repo-mapping fast.
- **Pipeline-style work.** Phases 1–7 of the plan were tackled in order with checkable
  outputs at each step (FAISS index built → ntotal printed; FastAPI imports clean →
  TestClient smoke test; eval runs → JSON + chart). Easy to verify each phase before
  moving on.
- **Memory of prior preferences.** The agent carried project-wide rules across turns —
  pinned `==` versions in `requirements.txt`, the `ninthc` venv, the matplotlib chart
  style — without me restating them.

## What didn't work

- **Model EOL surprise.** The first ingest run failed because NVIDIA had end-of-lifed the
  user-specified embed model the previous day. The agent had to discover this from the
  410 response, probe the live model catalog, find the successor, and update the code —
  ~3 extra tool calls and one wasted ingest run. A pre-flight `GET /v1/models` check
  would have caught this earlier.
- **Plan-mode UI sync.** Early in the planning phase the user kept seeing references to
  GCP / Vertex AI / Groq that the agent had already removed from the plan file. Likely a
  UI staleness issue, but the agent had to triple-check and write defensive "no
  references to X anywhere" clauses.
- **Free-tier rate limits leaking into iteration.** Running the 20-question eval against
  the local API hit NVIDIA's 429s mid-way through. The agent had to add retries +
  spacing and re-run, costing a few minutes. A smaller "smoke" subset for dev iteration
  would have been cheaper.
- **`pip` wheel availability on Python 3.14.** The user's venv is on Python 3.14
  (bleeding edge). Several initially-pinned versions (`tiktoken`, `pyarrow`,
  `pydantic-core`) didn't have 3.14 wheels and had to be bumped to current latest.
  Standard `==` pinning workflow against an older Python would have been smoother.

## What I'd do differently

- **Build a 5-question "smoke" eval.** A `evaluation/smoke.json` with 5 representative
  questions (one per category) would let dev iteration cost ~15 NVIDIA calls instead of
  ~60.
- **Cache the NVIDIA model availability check.** A tiny `pipeline/check_models.py` that
  pings each configured model ID and fails loudly would catch EOL events at CI time
  rather than at runtime.
- **Stream `/chat` responses.** Latency p95 of 55 s on a free-tier rate-limited backend
  makes the UI feel broken. SSE streaming (already noted as Sprint-2 priority in the
  plan) would give immediate "tokens appearing" feedback. ~30 lines of code with the
  OpenAI SDK's streaming mode.

## Roughly how much was human vs AI

The user defined: the corpus, the rubric, the deployment topology (Render+Vercel+FAISS),
the model choices (NVIDIA NIM stack), and every architectural pivot during the four plan
revisions. The agent wrote the code, ran the tests, and produced the docs.

Estimated split for the RAG layer:
- **Human (architecture, decisions, review):** all design tradeoffs, ~5–10 inline
  comments per plan revision that drove rewrites
- **AI (implementation):** all Python, all JSX/JS, all YAML, all Markdown
- **Hand edits to AI output:** essentially zero — the agent's first-pass code was used
  as-is in every phase.
