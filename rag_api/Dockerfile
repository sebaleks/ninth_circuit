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

# Copy code + index data (Render's buildCommand will have run `git lfs pull`)
COPY rag_api/ /app/rag_api/
COPY data/   /app/data/

EXPOSE 8000

# Render sets $PORT at runtime; default 8000 for local docker run
CMD ["sh", "-c", "uvicorn rag_api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
