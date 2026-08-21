"""ExamForge API entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import legacy, routes
from app.core import config, store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("examforge")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_directories()
    store.init_db()
    logger.info("ExamForge ready. data=%s chroma=%s",
                config.DATABASE_PATH, config.CHROMA_PATH)
    if not config.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY is not set; answer generation will fail.")
    yield


app = FastAPI(
    title="ExamForge API",
    description=(
        "Retrieval-augmented study assistant. Hybrid dense + BM25 retrieval with "
        "Multi-Query and HyDE expansion, cross-encoder reranking, and page-level "
        "citations derived from the retrieved chunks."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_origin_regex=config.CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)
app.include_router(legacy.router)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {
        "status": "healthy",
        "version": "2.0.0",
        "models": {
            "embedding": config.EMBEDDING_MODEL,
            "utility": config.UTILITY_MODEL,
            "generation": config.GENERATION_MODEL,
        },
        **store.stats(),
    }


@app.get("/", tags=["meta"])
async def root() -> dict:
    return {"service": "ExamForge API", "docs": "/docs", "health": "/health"}
