FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the first request after a deploy
# does not pay a cold download. Reranker weights are fetched on first use.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY . .

# Mount a persistent volume here on Northflank so vectors and conversation
# history survive redeploys.
ENV DATA_DIR=/app/data \
    CHROMA_PATH=/app/data/chroma \
    DATABASE_PATH=/app/data/examforge.db
RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Single worker is deliberate: the embedding model and reranker are loaded in
# process, and SQLite + Chroma are local files.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
