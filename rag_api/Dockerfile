# rag-api: FastAPI backend for Ninth Circuit asylum-case RAG
#
# Built on Render free tier. The build clones the repo (LFS pull happens via
# render.yaml's buildCommand) so data/index.faiss + data/metadata.parquet are
# materialized before the container starts.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install minimal build deps (faiss-cpu wheels are pre-built; this is for resilience)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git-lfs \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Install Python deps first for layer caching
COPY rag_api/requirements.txt /app/rag_api/requirements.txt
RUN pip install -r /app/rag_api/requirements.txt

# Bake the local embedder (e5-small-v2) into the image so Render cold starts
# (free tier sleeps after 15 min) don't re-download it on every wake. ~130MB.
# Only exercised when VECTOR_STORE/config selects the local embedder; harmless
# otherwise. HF_HOME is set so the runtime reads from this baked cache.
ENV HF_HOME=/app/.hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-small-v2', device='cpu')"

# Copy code + index data (Render's buildCommand will have run `git lfs pull`)
COPY rag_api/ /app/rag_api/
COPY data/   /app/data/

EXPOSE 8000

# Render sets $PORT at runtime; default 8000 for local docker run
CMD ["sh", "-c", "uvicorn rag_api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
