from pathlib import Path
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32,
    },
    show_progress=True,
)


def load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF, split its text into chunks, and attach useful metadata.
    """

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []

    for page in pages:
        page_content = page.page_content.strip()

        if not page_content:
            continue

        page_chunks = splitter.split_text(page_content)

        for i, chunk in enumerate(page_chunks):
            chunk = chunk.strip()

            if not chunk:
                continue

            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": Path(file_path).name,
                        "page": page.metadata.get("page", 0) + 1,
                        "chunk_index": i,
                    },
                )
            )

    if not chunks:
        raise ValueError(
            "No extractable text was found in the PDF. "
            "The PDF may be scanned, image-only, empty, or unsupported."
        )

    return chunks


def ingest_pdf(
    file_path: str,
    session_id: str,
    original_filename: str | None = None,
):
    """
    Ingest a PDF into a session-specific Chroma collection.
    """

    chunks = load_pdf(file_path)

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=session_id,
            persist_directory=CHROMA_PATH,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to store PDF embeddings: {exc}") from exc

    return {
        "session_id": session_id,
        "source": original_filename or Path(file_path).name,
        "chunks_stored": len(chunks),
    }
