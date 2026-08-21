"""Persistence for sessions, documents, conversations and messages.

Replaces the previous in-memory ``memory_store`` / ``uploaded_sessions`` dicts,
which lost every active session on restart while leaving orphaned Chroma
collections on disk forever.

Uses stdlib ``sqlite3`` deliberately: it adds no dependency, it is a single file
that sits on the same mounted volume as the vector store, and it is more than
sufficient for this workload. A connection is opened per operation, which is
safe under FastAPI's threadpool and avoids cross-thread connection sharing.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from app.core import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    filename      TEXT NOT NULL,
    page_count    INTEGER NOT NULL DEFAULT 0,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'ready',
    error         TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    title       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id               TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    citations        TEXT NOT NULL DEFAULT '[]',
    grounded         INTEGER NOT NULL DEFAULT 1,
    interpreted_as   TEXT,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_session
    ON documents(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session
    ON conversations(session_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


def new_id() -> str:
    """Public id generator for callers that need an id before insert."""
    return _new_id()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.DATABASE_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    config.ensure_directories()
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)


# --- Sessions ----------------------------------------------------------------


def create_session() -> str:
    session_id = _new_id()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
            (session_id, _now()),
        )
    return session_id


def session_exists(session_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return row is not None


def ensure_session(session_id: str) -> None:
    """Adopt a session id we have not seen before.

    Lets a client that still holds an id from a previous deployment keep working
    instead of hitting a dead end, and keeps the legacy endpoints honest.
    """
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)",
            (session_id, _now()),
        )


# --- Documents ---------------------------------------------------------------


def add_document(
    session_id: str,
    filename: str,
    page_count: int,
    chunk_count: int,
    size_bytes: int,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Register an ingested document.

    ``doc_id`` is supplied by the caller so the row id matches the ``doc_id``
    embedded in the document's vector metadata, which is what makes per-document
    deletion possible.
    """
    doc_id = doc_id or _new_id()
    created = _now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO documents
               (id, session_id, filename, page_count, chunk_count,
                size_bytes, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'ready', ?)""",
            (doc_id, session_id, filename, page_count, chunk_count,
             size_bytes, created),
        )
    return {
        "id": doc_id,
        "session_id": session_id,
        "filename": filename,
        "page_count": page_count,
        "chunk_count": chunk_count,
        "size_bytes": size_bytes,
        "status": "ready",
        "created_at": created,
    }


def list_documents(session_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, session_id, filename, page_count, chunk_count,
                      size_bytes, status, error, created_at
               FROM documents WHERE session_id = ? ORDER BY created_at""",
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_document(doc_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_document(doc_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


def has_documents(session_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE session_id = ? AND status = 'ready' LIMIT 1",
            (session_id,),
        ).fetchone()
    return row is not None


# --- Conversations -----------------------------------------------------------


def create_conversation(session_id: str, title: str | None = None) -> dict[str, Any]:
    conv_id = _new_id()
    now = _now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO conversations (id, session_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (conv_id, session_id, title, now, now),
        )
    return {
        "id": conv_id,
        "session_id": session_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
    }


def list_conversations(session_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT c.id, c.session_id, c.title, c.created_at, c.updated_at,
                      (SELECT COUNT(*) FROM messages m
                       WHERE m.conversation_id = c.id) AS message_count
               FROM conversations c
               WHERE c.session_id = ?
               ORDER BY c.updated_at DESC""",
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conv_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_conversation(conv_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


def rename_conversation(conv_id: str, title: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), conv_id),
        )


def _touch_conversation(conn: sqlite3.Connection, conv_id: str) -> None:
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conv_id)
    )


# --- Messages ----------------------------------------------------------------


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    citations: list[dict[str, Any]] | None = None,
    grounded: bool = True,
    interpreted_as: str | None = None,
) -> dict[str, Any]:
    msg_id = _new_id()
    created = _now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO messages
               (id, conversation_id, role, content, citations, grounded,
                interpreted_as, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg_id, conversation_id, role, content,
             json.dumps(citations or []), 1 if grounded else 0,
             interpreted_as, created),
        )
        _touch_conversation(conn, conversation_id)

        # First user message becomes the conversation title.
        if role == "user":
            row = conn.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is not None and not row["title"]:
                conn.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?",
                    (_derive_title(content), conversation_id),
                )

    return {
        "id": msg_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "citations": citations or [],
        "grounded": grounded,
        "interpreted_as": interpreted_as,
        "created_at": created,
    }


def list_messages(conversation_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, conversation_id, role, content, citations, grounded,
                      interpreted_as, created_at
               FROM messages WHERE conversation_id = ? ORDER BY created_at""",
            (conversation_id,),
        ).fetchall()

    messages = []
    for row in rows:
        item = dict(row)
        try:
            item["citations"] = json.loads(item["citations"])
        except (TypeError, ValueError):
            item["citations"] = []
        item["grounded"] = bool(item["grounded"])
        messages.append(item)
    return messages


def delete_message(message_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))


def delete_messages_from(conversation_id: str, message_id: str) -> None:
    """Drop a message and everything after it. Used by regenerate."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT created_at FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if row is None:
            return
        conn.execute(
            """DELETE FROM messages
               WHERE conversation_id = ? AND created_at >= ?""",
            (conversation_id, row["created_at"]),
        )


def get_memory(conversation_id: str, window: int) -> list[dict[str, str]]:
    """Recent turns for question condensation.

    Returns the user's *original* wording, not the rewritten form, so history
    never drifts from what was actually said.
    """
    messages = list_messages(conversation_id)
    trimmed = messages[-window:] if window > 0 else messages
    return [{"role": m["role"], "content": m["content"]} for m in trimmed]


def _derive_title(text: str, limit: int = 60) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= limit:
        return cleaned or "New conversation"
    return cleaned[:limit].rsplit(" ", 1)[0] + "…"


# --- Stats -------------------------------------------------------------------


def stats() -> dict[str, int]:
    with _connect() as conn:
        def count(table: str) -> int:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        return {
            "sessions": count("sessions"),
            "documents": count("documents"),
            "conversations": count("conversations"),
            "messages": count("messages"),
        }
