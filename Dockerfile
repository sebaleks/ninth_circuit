FROM python:3.12-slim

WORKDIR /app

# onnxruntime (rag_api.onnx_embedder, the torch-free e5 path) links libgomp.so.1;
# python:3.12-slim doesn't ship the OpenMP runtime, so install it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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