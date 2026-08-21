"""PDF ingestion: extract, page-aware chunking, embed, persist to Chroma."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core import config

logger = logging.getLogger(__name__)

# Re-exported for backwards compatibility with existing imports.
CHROMA_PATH = config.CHROMA_PATH

embedding_model = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)


def get_vectorstore(session_id: str) -> Chroma:
    """One Chroma collection per session, persisted to disk."""
    return Chroma(
        collection_name=session_id,
        embedding_function=embedding_model,
        persist_directory=config.CHROMA_PATH,
    )


def load_pdf(
    file_path: str,
    display_name: str | None = None,
    doc_id: str | None = None,
) -> tuple[list[Document], int]:
    """Split a PDF into page-attributed chunks.

    ``display_name`` is the filename the user actually uploaded. The previous
    implementation used ``Path(file_path).name``, which was the NamedTemporaryFile
    path, so every citation reported a meaningless name like ``tmpq8x_31ab.pdf``.
    """
    source_name = display_name or Path(file_path).name

    loader = PyPDFLoader(file_path)
    pages = loader.load()
    page_count = len(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []

    for page in pages:
        page_content = page.page_content.strip()
        if not page_content:
            continue

        page_number = page.metadata.get("page", 0) + 1

        for index, chunk in enumerate(splitter.split_text(page_content)):
            chunk = chunk.strip()
            if not chunk:
                continue

            metadata: dict[str, Any] = {
                "source": source_name,
                "page": page_number,
                "chunk_index": index,
            }
            if doc_id:
                metadata["doc_id"] = doc_id

            chunks.append(Document(page_content=chunk, metadata=metadata))

    if not chunks:
        raise ValueError(
            "No extractable text was found in this PDF. It may be scanned, "
            "image-only, empty, or password protected."
        )

    return chunks, page_count


def ingest_pdf(
    file_path: str,
    session_id: str,
    original_filename: str | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Ingest a PDF into the session's Chroma collection.

    Adding a second document to an existing session appends to the same
    collection, so a session can hold several documents and retrieval spans
    all of them.
    """
    chunks, page_count = load_pdf(file_path, original_filename, doc_id)

    try:
        vectorstore = get_vectorstore(session_id)
        vectorstore.add_documents(chunks)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client
        logger.exception("Failed to embed PDF for session %s", session_id)
        raise RuntimeError(f"Failed to store document embeddings: {exc}") from exc

    invalidate_cache(session_id)

    return {
        "session_id": session_id,
        "doc_id": doc_id,
        "source": original_filename or Path(file_path).name,
        "page_count": page_count,
        "chunks_stored": len(chunks),
    }


def delete_document_vectors(session_id: str, doc_id: str) -> None:
    """Remove one document's chunks, leaving other documents intact."""
    try:
        vectorstore = get_vectorstore(session_id)
        vectorstore._collection.delete(where={"doc_id": doc_id})
    except Exception:  # noqa: BLE001
        logger.exception("Failed to delete vectors for doc %s", doc_id)
        raise
    invalidate_cache(session_id)


def delete_session_vectors(session_id: str) -> None:
    try:
        get_vectorstore(session_id).delete_collection()
    except Exception:  # noqa: BLE001
        logger.warning("Could not delete collection %s", session_id, exc_info=True)
    invalidate_cache(session_id)


# --- Chunk cache -------------------------------------------------------------
# The BM25 half of hybrid retrieval needs the full chunk set. Previously every
# question dumped the entire collection from Chroma and rebuilt the index from
# scratch. Caching per session removes that cost from the hot path; the cache is
# invalidated whenever the session's documents change.

_chunk_cache: dict[str, list[Document]] = {}


def invalidate_cache(session_id: str) -> None:
    _chunk_cache.pop(session_id, None)


def get_all_chunks(session_id: str) -> list[Document]:
    cached = _chunk_cache.get(session_id)
    if cached is not None:
        return cached

    results = get_vectorstore(session_id).get()
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    chunks = [
        Document(
            page_content=text,
            metadata=metadatas[i] if i < len(metadatas) else {},
        )
        for i, text in enumerate(documents)
    ]

    _chunk_cache[session_id] = chunks
    return chunks
