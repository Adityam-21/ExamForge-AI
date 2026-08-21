from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    session_id: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    page_count: int
    chunk_count: int
    size_bytes: int
    status: str
    created_at: str


class ConversationResponse(BaseModel):
    id: str
    session_id: str
    title: str | None = None
    created_at: str
    updated_at: str
    message_count: int = 0


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    grounded: bool = True
    interpreted_as: str | None = None
    created_at: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    # When regenerating, the assistant message to replace. Everything from that
    # message onward is discarded and the preceding user turn is reused.
    replace_message_id: str | None = None


class SessionStateResponse(BaseModel):
    """Everything the client needs to rehydrate after a refresh."""

    session_id: str
    documents: list[DocumentResponse]
    conversations: list[ConversationResponse]


class LegacyAskRequest(BaseModel):
    session_id: str
    question: str


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
