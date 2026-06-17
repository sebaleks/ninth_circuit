FROM python:3.12-slim

WORKDIR /app

# onnxruntime (rag_api.onnx_embedder, the torch-free e5 path) links libgomp.so.1;
# python:3.12-slim doesn't ship the OpenMP runtime, so install it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the confidence-indicator models (L-6 cross-encoder + Qdrant/bm25 sparse) INTO the image
# so they are not downloaded from HuggingFace on the first request — no cold-start stall, no runtime
# HF dependency. FASTEMBED_CACHE_PATH is a persistent in-image dir shared by this build-time fetch
# AND the runtime (the ENV persists into the container). The model names must match the defaults in
# rag_api/cross_encoder.py (CE_MODEL) and rag_api/sparse_store.py (BM25_MODEL); override + re-bake if
# you change them. ~88 MB on disk; loaded into RAM only when CONFIDENCE_ENABLED/BM25_BACKEND=qdrant.
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
RUN python -c "\
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
ce = TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2'); list(ce.rerank('warm', ['warm passage'])); \
from fastembed import SparseTextEmbedding; \
sp = SparseTextEmbedding(model_name='Qdrant/bm25'); list(sp.embed(['warm passage'])); \
print('pre-baked: L-6 cross-encoder + Qdrant/bm25')"

COPY lib/ lib/
COPY pipeline/ pipeline/
COPY rag_api/ rag_api/
COPY data/ data/
COPY cloud/main.py .
COPY cloud/run_fetch.py .
COPY cloud/run_classify.py .
COPY cloud/run_extract.py .
COPY cloud/run_backfill.py .
COPY cloud/run_classify_batch.py .
COPY cloud/run_qa.py .
COPY cloud/run_backup.py .
COPY cloud/entrypoint.py .

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn rag_api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]