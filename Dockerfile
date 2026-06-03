FROM python:3.12-slim

WORKDIR /app

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