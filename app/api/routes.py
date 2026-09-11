"""ExamForge API.

Resource-oriented and versioned under /api. A session owns documents (one Chroma
collection) and can hold several conversations, each with its own transcript.
That split is what makes "New conversation" possible without forcing a re-upload.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
from typing import Any, AsyncIterator

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    AskRequest,
    ConversationResponse,
    DocumentResponse,
    MessageResponse,
    RenameRequest,
    SessionResponse,
    SessionStateResponse,
)
from app.core import config, store
from app.services import ingestion
from app.services.agent import run_pipeline

from langsmith import traceable


@traceable(
    name="examforge_ask",
    run_type="chain",
    metadata={
        "generation_model": config.GENERATION_MODEL,
        "utility_model": config.UTILITY_MODEL,
        "retriever_k": config.RETRIEVER_K,
        "rerank_top_n": config.RERANK_TOP_N,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embedding_model": config.EMBEDDING_MODEL,
        "rerank_enabled": True,
    },
)
def _traced_pipeline(session_id, question, memory, emit):
    """Tagged wrapper so runs are comparable across configurations.

    Untagged runs cannot be filtered by config later, which makes them
    useless for before/after comparison.
    """
    return run_pipeline(session_id, question, memory, emit)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# --- Helpers -----------------------------------------------------------------


def _require_session(session_id: str) -> None:
    if not store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")


def _require_conversation(conversation_id: str) -> dict[str, Any]:
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def _document_payload(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": doc["id"],
        "filename": doc["filename"],
        "page_count": doc["page_count"],
        "chunk_count": doc["chunk_count"],
        "size_bytes": doc["size_bytes"],
        "status": doc["status"],
        "created_at": doc["created_at"],
    }


# --- Sessions ----------------------------------------------------------------


@router.post("/sessions", response_model=SessionResponse)
async def create_session() -> dict[str, str]:
    return {"session_id": store.create_session()}


@router.get("/sessions/{session_id}", response_model=SessionStateResponse)
async def get_session_state(session_id: str) -> dict[str, Any]:
    """Rehydrate a client on load.

    Unknown ids are adopted rather than rejected, so a client holding an id from
    a previous deployment recovers instead of hitting a dead end.
    """
    store.ensure_session(session_id)
    return {
        "session_id": session_id,
        "documents": [_document_payload(d) for d in store.list_documents(session_id)],
        "conversations": store.list_conversations(session_id),
    }


# --- Documents ---------------------------------------------------------------


@router.get("/sessions/{session_id}/documents", response_model=list[DocumentResponse])
async def list_documents(session_id: str) -> list[dict[str, Any]]:
    _require_session(session_id)
    return [_document_payload(d) for d in store.list_documents(session_id)]


@router.post("/sessions/{session_id}/documents", response_model=DocumentResponse)
async def upload_document(
    session_id: str, file: UploadFile = File(...)
) -> dict[str, Any]:
    store.ensure_session(session_id)

    filename = file.filename or "document.pdf"
    is_pdf = file.content_type == "application/pdf" or filename.lower().endswith(".pdf")
    if not is_pdf:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    if len(payload) > config.MAX_UPLOAD_BYTES:
        limit_mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"This PDF is too large. The maximum size is {limit_mb} MB.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
        handle.write(payload)
        temp_path = handle.name

    doc_id = store.new_id()

    try:
        result = await asyncio.to_thread(
            ingestion.ingest_pdf, temp_path, session_id, filename, doc_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingestion failed for session %s", session_id)
        raise HTTPException(
            status_code=500, detail="Could not process this PDF. Please try again."
        ) from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    document = store.add_document(
        session_id=session_id,
        filename=filename,
        page_count=result["page_count"],
        chunk_count=result["chunks_stored"],
        size_bytes=len(payload),
        doc_id=doc_id,
    )

    return _document_payload(document)


@router.delete(
    "/sessions/{session_id}/documents/{doc_id}",
    status_code=204,
    response_class=Response,
)
async def delete_document(session_id: str, doc_id: str) -> Response:
    _require_session(session_id)
    document = store.get_document(doc_id)
    if document is None or document["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        await asyncio.to_thread(ingestion.delete_document_vectors, session_id, doc_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail="Could not remove this document."
        ) from exc

    store.delete_document(doc_id)
    return Response(status_code=204)


# --- Conversations -----------------------------------------------------------


@router.get(
    "/sessions/{session_id}/conversations",
    response_model=list[ConversationResponse],
)
async def list_conversations(session_id: str) -> list[dict[str, Any]]:
    _require_session(session_id)
    return store.list_conversations(session_id)


@router.post(
    "/sessions/{session_id}/conversations", response_model=ConversationResponse
)
async def create_conversation(session_id: str) -> dict[str, Any]:
    store.ensure_session(session_id)
    return store.create_conversation(session_id)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
async def list_messages(conversation_id: str) -> list[dict[str, Any]]:
    _require_conversation(conversation_id)
    return store.list_messages(conversation_id)


@router.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def rename_conversation(
    conversation_id: str, body: RenameRequest
) -> dict[str, Any]:
    _require_conversation(conversation_id)
    store.rename_conversation(conversation_id, body.title.strip())
    return store.get_conversation(conversation_id) | {"message_count": 0}


@router.delete(
    "/conversations/{conversation_id}",
    status_code=204,
    response_class=Response,
)
async def delete_conversation(conversation_id: str) -> Response:
    _require_conversation(conversation_id)
    store.delete_conversation(conversation_id)
    return Response(status_code=204)


# --- Ask ---------------------------------------------------------------------


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _prepare_turn(
    conversation_id: str, body: AskRequest
) -> tuple[str, list[dict[str, str]]]:
    """Validate the turn and return (session_id, memory)."""
    conversation = _require_conversation(conversation_id)
    session_id = conversation["session_id"]

    if not store.has_documents(session_id):
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF before asking a question.",
        )

    if body.replace_message_id:
        # Regenerating: discard the old answer and reuse the existing user turn.
        store.delete_messages_from(conversation_id, body.replace_message_id)
        memory = store.get_memory(conversation_id, config.MEMORY_WINDOW)
        # The user turn is already persisted, so exclude it from memory.
        if memory and memory[-1]["role"] == "user":
            memory = memory[:-1]
    else:
        memory = store.get_memory(conversation_id, config.MEMORY_WINDOW)
        store.add_message(conversation_id, "user", body.question.strip())

    return session_id, memory


@router.post("/conversations/{conversation_id}/ask/stream")
async def ask_stream(
    conversation_id: str, body: AskRequest, request: Request
) -> StreamingResponse:
    """Stream real pipeline stages, then the answer, then verified sources."""
    session_id, memory = _prepare_turn(conversation_id, body)
    question = body.question.strip()

    async def generate() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        collected: dict[str, Any] = {"answer": "", "interpreted_as": None}
        completed = False

        def emit(event: str, payload: dict[str, Any]) -> None:
            if event == "token":
                collected["answer"] += payload.get("text", "")
            elif event == "interpretation":
                collected["interpreted_as"] = payload.get("question")
            loop.call_soon_threadsafe(queue.put_nowait, (event, payload))

        def worker() -> None:
            try:
                result = _traced_pipeline(session_id, question, memory, emit)
                loop.call_soon_threadsafe(queue.put_nowait, ("__done__", result))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Pipeline failed")
                loop.call_soon_threadsafe(queue.put_nowait, ("__error__", exc))

        threading.Thread(target=worker, daemon=True).start()

        try:
            yield _sse("start", {"conversation_id": conversation_id})

            while True:
                event, payload = await queue.get()

                if event == "__error__":
                    yield _sse(
                        "error",
                        {
                            "message": "Something went wrong while answering. "
                            "Please try again."
                        },
                    )
                    return

                if event == "__done__":
                    completed = True
                    message = store.add_message(
                        conversation_id,
                        "assistant",
                        payload.get("answer") or "",
                        citations=payload.get("citations") or [],
                        grounded=bool(payload.get("grounded")),
                        interpreted_as=collected["interpreted_as"],
                    )
                    yield _sse(
                        "done",
                        {
                            "message_id": message["id"],
                            "grounded": message["grounded"],
                            "citations": message["citations"],
                            "created_at": message["created_at"],
                        },
                    )
                    return

                yield _sse(event, payload)

        finally:
            # The client stopped generation or disconnected: keep whatever was
            # produced so the transcript matches what the user saw.
            if not completed and collected["answer"].strip():
                store.add_message(
                    conversation_id,
                    "assistant",
                    collected["answer"].strip(),
                    citations=[],
                    grounded=False,
                    interpreted_as=collected["interpreted_as"],
                )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/conversations/{conversation_id}/ask", response_model=MessageResponse)
async def ask(conversation_id: str, body: AskRequest) -> dict[str, Any]:
    """Non-streaming fallback for clients that cannot consume SSE."""
    session_id, memory = _prepare_turn(conversation_id, body)

    interpreted: dict[str, Any] = {"question": None}

    def emit(event: str, payload: dict[str, Any]) -> None:
        if event == "interpretation":
            interpreted["question"] = payload.get("question")

    try:
        result = await asyncio.to_thread(
            _traced_pipeline, session_id, body.question.strip(), memory, emit
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while answering. Please try again.",
        ) from exc

    return store.add_message(
        conversation_id,
        "assistant",
        result.get("answer") or "",
        citations=result.get("citations") or [],
        grounded=bool(result.get("grounded")),
        interpreted_as=interpreted["question"],
    )
