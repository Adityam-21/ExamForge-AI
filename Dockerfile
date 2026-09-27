FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    MODEL_CACHE_DIR=/app/.cache

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake both models into the image so a cold start (Render Free spins down after
# 15 min idle) does not also pay a model download.
RUN python -c "from fastembed import TextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/.cache/fastembed')" \
 && python -c "from flashrank import Ranker; \
Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/app/.cache/flashrank')"

COPY . .

# Render Free has an ephemeral filesystem: vectors and conversation history
# reset on redeploy, restart or spin-down. Unknown session ids are adopted by
# the API, so clients recover with an empty session instead of erroring.
ENV DATA_DIR=/app/data \
    CHROMA_PATH=/app/data/chroma \
    DATABASE_PATH=/app/data/examforge.db
RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-8000}/health || exit 1

# Render injects $PORT; local Docker falls back to 8000.
# Single worker is deliberate: models are loaded in process and SQLite + Chroma
# are local files.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
