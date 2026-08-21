"""Original v1 endpoints, preserved.

The previous frontend and the published API docs use ``/session``, ``/upload``
and ``/ask``. Those contracts still work and return the same shapes; they are
implemented on top of the new resource API so there is one pipeline, not two.

Each legacy session lazily gets a single implicit conversation, which is what the
old flat ``session_id`` model effectively was.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api import routes
from app.api.schemas import AskRequest, LegacyAskRequest
from app.core import store

router = APIRouter(tags=["legacy"])


def _implicit_conversation(session_id: str) -> str:
    conversations = store.list_conversations(session_id)
    if conversations:
        return conversations[0]["id"]
    return store.create_conversation(session_id, title="Session")["id"]


@router.post("/session", deprecated=True)
async def create_session() -> dict[str, str]:
    return {"session_id": store.create_session()}


@router.post("/upload", deprecated=True)
async def upload(session_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    document = await routes.upload_document(session_id, file)
    return {
        "session_id": session_id,
        "source": document["filename"],
        "chunks_stored": document["chunk_count"],
    }


@router.post("/ask", deprecated=True)
async def ask(request: LegacyAskRequest) -> dict[str, Any]:
    if not store.session_exists(request.session_id):
        raise HTTPException(
            status_code=404, detail="Session not found, upload a PDF first"
        )
    if not store.has_documents(request.session_id):
        raise HTTPException(
            status_code=400, detail="No PDF uploaded for this session"
        )

    conversation_id = _implicit_conversation(request.session_id)
    message = await routes.ask(
        conversation_id, AskRequest(question=request.question)
    )

    return {
        "answer": message["content"],
        "citations": message["citations"],
    }
