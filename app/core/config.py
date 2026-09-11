"""Central configuration. Everything deployment-sensitive is env-driven."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# --- Secrets -----------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Storage -----------------------------------------------------------------
# On Northflank both of these should live on the same mounted volume so that
# vectors and conversation state survive a redeploy together.

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHROMA_PATH = os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma"))
DATABASE_PATH = os.getenv("DATABASE_PATH", str(DATA_DIR / "examforge.db"))

# --- Models ------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Small, fast model for query transformation (multi-query, HyDE, condensation).
UTILITY_MODEL = os.getenv("UTILITY_MODEL", "llama-3.1-8b-instant")

# Larger model for the final grounded answer.
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-oss-120b")

# --- Ingestion ---------------------------------------------------------------

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024

# --- Retrieval ---------------------------------------------------------------

RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

# FlashRank measurably degrades retrieval on the eval corpus
# (MRR 0.766 -> 0.399, n=39; see docs/EVALUATION.md). Disabled by default,
# kept behind a flag so the ablation stays reproducible.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"

# Chunks scoring below this after reranking are treated as too weak to ground
# an answer. Surfaced to the client as low confidence rather than hidden.
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.05"))

# How many prior turns feed question condensation.
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "6"))

# --- API ---------------------------------------------------------------------

CORS_ORIGINS = _csv_env(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,https://exam-forge-ai.vercel.app",
)

# Allow any Vercel preview deployment of this project without listing each one.
CORS_ORIGIN_REGEX = os.getenv(
    "CORS_ORIGIN_REGEX", r"https://exam-forge-ai.*\.vercel\.app"
)


def ensure_directories() -> None:
    """Create storage directories at startup so first write cannot fail."""
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
