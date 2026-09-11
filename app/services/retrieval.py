"""Retrieval pipeline.

Four strategies run against the session's Chroma collection and their results
are fused and deduplicated:

1. Dense       - embedding similarity over the collection.
2. Sparse      - BM25 keyword matching over the same chunks.
3. Multi-Query - an LLM rewrites the question into several phrasings.
4. HyDE        - an LLM drafts a hypothetical textbook answer and we embed that
                 instead of the question, closing the vocabulary gap between
                 casual questions and academic prose.

Dense + sparse are fused with reciprocal rank fusion via EnsembleRetriever.

RE-RANKING IS DISABLED BY DEFAULT. A FlashRank cross-encoder reorder was
measured against a 39-question chunk-anchored golden set and found to DEGRADE
retrieval on this corpus: MRR 0.766 -> 0.399, hit-rate@1 0.641 -> 0.231
(replicated across three runs; see docs/EVALUATION.md). It is kept behind
config.RERANK_ENABLED so the ablation stays reproducible and so the regression
suite can use it as a known-bad arm.

Two behaviours matter for the product:

* ``on_stage`` reports genuine pipeline progress so the UI can show what the
  system is actually doing instead of an opaque spinner.
* Each strategy is individually fault-tolerant. A transient LLM failure in
  Multi-Query or HyDE degrades retrieval quality but no longer fails the
  request outright.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable

from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from app.core import config
from app.services.ingestion import get_all_chunks, get_vectorstore

logger = logging.getLogger(__name__)

StageCallback = Callable[[str, str], None]


@lru_cache(maxsize=1)
def get_utility_llm() -> ChatGroq:
    """Small fast model for query transformation only.

    Built lazily and cached so a missing API key surfaces as a handled request
    error rather than crashing at import. The previous module also declared a
    `llama-3.1-70b-versatile` client that was never used and names a model Groq
    has since decommissioned; it has been removed.
    """
    return ChatGroq(
        model=config.UTILITY_MODEL,
        temperature=0,
        api_key=config.GROQ_API_KEY,
    )


HYDE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "Generate a hypothetical answer to the following question. This answer "
        "will be used to search a document database, not shown to the user. "
        "Write it as if it came directly from a textbook or academic document. "
        "Be concise, factual, and specific.\n\n"
        "Question: {question}\nHypothetical Answer:"
    ),
)


def _noop(stage: str, detail: str = "") -> None:
    return None


# --- Individual strategies ---------------------------------------------------


def _dense_and_sparse(
    session_id: str, question: str, chunks: list[Document]
) -> list[Document]:
    """Hybrid retrieval: BM25 + dense, fused with reciprocal rank fusion."""
    semantic = get_vectorstore(session_id).as_retriever(
        search_kwargs={"k": config.RETRIEVER_K}
    )

    if not chunks:
        return semantic.invoke(question)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = config.RETRIEVER_K

    ensemble = EnsembleRetriever(retrievers=[bm25, semantic], weights=[0.5, 0.5], c=60)
    return ensemble.invoke(question)


def _multi_query(session_id: str, question: str) -> list[Document]:
    base = get_vectorstore(session_id).as_retriever(
        search_kwargs={"k": config.RETRIEVER_K}
    )
    retriever = MultiQueryRetriever.from_llm(retriever=base, llm=get_utility_llm())
    return retriever.invoke(question)


def _hyde(session_id: str, question: str) -> list[Document]:
    chain = HYDE_PROMPT | get_utility_llm()
    hypothetical = chain.invoke({"question": question}).content
    base = get_vectorstore(session_id).as_retriever(
        search_kwargs={"k": config.RETRIEVER_K}
    )
    return base.invoke(hypothetical)


def _safe(label: str, fn, *args) -> list[Document]:
    """Run one strategy, tolerating its failure."""
    try:
        return fn(*args) or []
    except Exception:  # noqa: BLE001
        logger.warning("Retrieval strategy '%s' failed", label, exc_info=True)
        return []


# --- Fusion ------------------------------------------------------------------


def deduplicate(chunks: list[Document]) -> list[Document]:
    seen: set[str] = set()
    unique: list[Document] = []
    for chunk in chunks:
        key = chunk.page_content.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


def _score_by_similarity(question: str, chunks: list[Document]) -> list[Document]:
    """Attach cosine similarity to chunks when the reranker is disabled.

    The abstention gate reads ``relevance_of()``, which originally only ever saw
    the FlashRank score. With RERANK_ENABLED=false nothing writes a score, the
    gate reads 0.0, and every question abstains - the pipeline looks healthy on
    retrieval metrics while answering nothing. Scoring locally against the same
    BGE embeddings keeps the gate meaningful and costs no API call.
    """
    if not chunks:
        return chunks
    try:
        import numpy as np

        from app.services.ingestion import embedding_model

        query = np.asarray(embedding_model.embed_query(question), dtype=float)
        docs = np.asarray(
            embedding_model.embed_documents([c.page_content for c in chunks]),
            dtype=float,
        )
        query = query / (np.linalg.norm(query) or 1.0)
        docs = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12)
        for chunk, score in zip(chunks, docs @ query):
            chunk.metadata["similarity_score"] = float(score)
    except Exception:  # noqa: BLE001
        logger.warning("Similarity scoring failed", exc_info=True)
    return chunks


def rerank(question: str, chunks: list[Document]) -> list[Document]:
    """Cross-encoder rerank. Falls back to first-stage order on failure.

    Disabled by default - see the module docstring for the measurement.
    """
    if not chunks:
        return []
    if not config.RERANK_ENABLED:
        return _score_by_similarity(question, chunks[: config.RERANK_TOP_N])
    try:
        compressor = FlashrankRerank(top_n=config.RERANK_TOP_N)
        return compressor.compress_documents(documents=chunks, query=question)
    except Exception:  # noqa: BLE001
        logger.warning("Reranking failed; using fusion order", exc_info=True)
        return _score_by_similarity(question, chunks[: config.RERANK_TOP_N])


def relevance_of(chunk: Document) -> float:
    """Relevance score for a chunk, used by the abstention gate and citations.

    Two sources, in order of preference:
      1. ``relevance_score`` written by the FlashRank reranker.
      2. ``similarity_score`` written by ``_score_by_similarity`` when the
         reranker is disabled.

    Note the two are on different scales - FlashRank emits a cross-encoder
    score, this emits cosine similarity - so MIN_RELEVANCE_SCORE means
    something slightly different in each mode. Recorded in EVALUATION.md.
    """
    for key in ("relevance_score", "similarity_score"):
        value = chunk.metadata.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


# --- Entry point -------------------------------------------------------------


def retrieve(
    session_id: str,
    question: str,
    on_stage: StageCallback | None = None,
) -> list[Document]:
    """Run the full pipeline and return the top chunks.

    Returns an empty list when the session has no indexed content or nothing
    matched, so the caller can report insufficient evidence rather than letting
    the model answer unsupported.
    """
    emit = on_stage or _noop

    emit("searching", "Searching your study material")
    chunks = get_all_chunks(session_id)
    if not chunks:
        return []

    hybrid = _safe("hybrid", _dense_and_sparse, session_id, question, chunks)

    emit("expanding", "Exploring related phrasings")
    multi = _safe("multi_query", _multi_query, session_id, question)
    hyde = _safe("hyde", _hyde, session_id, question)

    candidates = deduplicate(hybrid + multi + hyde)
    if not candidates:
        return []

    emit("reranking", f"Ranking {len(candidates)} candidate passages")
    ranked = rerank(question, candidates)

    return ranked
