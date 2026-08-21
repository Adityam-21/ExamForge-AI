"""API integration tests.

Exercises the real FastAPI app, routers and SQLite store. Only the two external
boundaries are stubbed: embedding/vector storage and the Groq LLM. Everything
else - routing, validation, SSE framing, persistence, regenerate truncation and
the legacy endpoint shims - is the production code path.

Run:  python tests/test_api_integration.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import types

DATA_DIR = "/tmp/examforge-itest"
shutil.rmtree(DATA_DIR, ignore_errors=True)
os.environ["DATA_DIR"] = DATA_DIR
os.environ["DATABASE_PATH"] = f"{DATA_DIR}/examforge.db"
os.environ["CHROMA_PATH"] = f"{DATA_DIR}/chroma"
os.environ["GROQ_API_KEY"] = "test-key"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Stub the ML boundary ----------------------------------------------------

def _stub_modules() -> None:
    class Doc:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(m, key, value)
        sys.modules[name] = m
        return m

    class Stub:
        def __init__(self, *a, **k):
            pass

        def __or__(self, other):
            return self

        def invoke(self, *a, **k):
            return self

    mod("langchain_core")
    mod("langchain_core.documents", Document=Doc)
    mod("langchain_core.prompts", PromptTemplate=Stub)
    mod("langchain_groq", ChatGroq=Stub)

    class SG(Stub):
        def add_node(self, *a, **k): pass
        def set_entry_point(self, *a): pass
        def add_edge(self, *a): pass
        def compile(self): return object()

    mod("langgraph")
    mod("langgraph.graph", StateGraph=SG, END="END")
    mod("langchain")
    mod("langchain.retrievers", EnsembleRetriever=Stub,
        ContextualCompressionRetriever=Stub)
    mod("langchain.retrievers.multi_query", MultiQueryRetriever=Stub)
    mod("langchain_community")
    mod("langchain_community.document_compressors", FlashrankRerank=Stub)
    mod("langchain_community.retrievers", BM25Retriever=Stub)
    mod("langchain_community.vectorstores", Chroma=Stub)
    mod("langchain_community.embeddings", HuggingFaceEmbeddings=lambda **k: None)
    mod("langchain_community.document_loaders", PyPDFLoader=Stub)
    mod("langchain_text_splitters", RecursiveCharacterTextSplitter=Stub)
    return Doc


Document = _stub_modules()

from fastapi.testclient import TestClient  # noqa: E402

from app.services import agent, ingestion  # noqa: E402

# --- Fake ingestion + pipeline ----------------------------------------------

INGESTED: list[dict] = []
DELETED: list[tuple[str, str]] = []


def fake_ingest(file_path, session_id, original_filename=None, doc_id=None):
    INGESTED.append({"session": session_id, "name": original_filename, "doc": doc_id})
    return {
        "session_id": session_id,
        "doc_id": doc_id,
        "source": original_filename,
        "page_count": 12,
        "chunks_stored": 87,
    }


ingestion.ingest_pdf = fake_ingest
ingestion.delete_document_vectors = lambda s, d: DELETED.append((s, d))

FAKE_CHUNKS = [
    Document(
        "Gauss's law relates electric flux to enclosed charge.",
        {"source": "physics.pdf", "page": 7, "chunk_index": 1,
         "doc_id": "d1", "relevance_score": 0.88},
    ),
    Document(
        "Faraday's law describes induced electromotive force.",
        {"source": "physics.pdf", "page": 9, "chunk_index": 0,
         "doc_id": "d1", "relevance_score": 0.31},
    ),
]

ANSWER_TOKENS = ["Gauss's law ", "states that flux ", "is proportional ",
                 "to enclosed charge [1]."]


def fake_run_pipeline(session_id, question, memory, emit=None):
    """Mimics the real graph's emission order."""
    emit = emit or (lambda e, p: None)
    emit("stage", {"stage": "understanding", "label": "Understanding your question"})
    if memory:
        emit("interpretation", {"question": f"{question} (in context)"})
    emit("stage", {"stage": "searching", "label": "Searching your study material"})
    emit("stage", {"stage": "reranking", "label": "Ranking the strongest evidence"})
    emit("sources", {"sources": agent.build_sources(FAKE_CHUNKS)})
    emit("stage", {"stage": "generating", "label": "Generating answer"})

    answer = ""
    for token in ANSWER_TOKENS:
        answer += token
        emit("token", {"text": token})

    citations = agent.build_sources(FAKE_CHUNKS, answer)
    return {
        "answer": answer,
        "citations": citations,
        "grounded": agent.is_grounded(answer, FAKE_CHUNKS, citations),
    }


import app.api.routes as routes  # noqa: E402

routes.run_pipeline = fake_run_pipeline
routes.ingestion = ingestion

import atexit  # noqa: E402
import contextlib  # noqa: E402

import main  # noqa: E402

# Entering the client as a context manager runs the app lifespan, so startup
# (directory creation + schema migration) is part of what gets tested.
_stack = contextlib.ExitStack()
client = _stack.enter_context(TestClient(main.app))
atexit.register(_stack.close)

# --- Harness ----------------------------------------------------------------

FAILURES: list[str] = []


def check(name, condition, extra=""):
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name} {extra}")


def parse_sse(text):
    events = []
    for frame in text.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        event, data = "message", []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            events.append((event, json.loads("\n".join(data))))
    return events


PDF = b"%PDF-1.4 fake bytes for upload validation"

print("\n--- health & session ---")
health = client.get("/health").json()
check("health reports healthy", health["status"] == "healthy")
check("health exposes model config", "generation" in health["models"])

session_id = client.post("/api/sessions").json()["session_id"]
check("session created", bool(session_id))

print("\n--- upload validation ---")
r = client.post(
    f"/api/sessions/{session_id}/documents",
    files={"file": ("notes.txt", b"hello", "text/plain")},
)
check("non-PDF rejected with 400", r.status_code == 400, r.text)

r = client.post(
    f"/api/sessions/{session_id}/documents",
    files={"file": ("empty.pdf", b"", "application/pdf")},
)
check("empty file rejected with 400", r.status_code == 400, r.text)

r = client.post(
    f"/api/sessions/{session_id}/documents",
    files={"file": ("huge.pdf", b"x" * (11 * 1024 * 1024), "application/pdf")},
)
check("oversized file rejected with 413", r.status_code == 413, r.text)

print("\n--- ask before upload ---")
conv = client.post(f"/api/sessions/{session_id}/conversations").json()
r = client.post(
    f"/api/conversations/{conv['id']}/ask", json={"question": "anything"}
)
check("ask without documents rejected with 400", r.status_code == 400, r.text)

print("\n--- upload success ---")
r = client.post(
    f"/api/sessions/{session_id}/documents",
    files={"file": ("physics.pdf", PDF, "application/pdf")},
)
check("PDF accepted", r.status_code == 200, r.text)
doc = r.json()
check("original filename preserved (not tempfile)", doc["filename"] == "physics.pdf")
check("ingest received real filename",
      INGESTED[-1]["name"] == "physics.pdf", str(INGESTED[-1]))
check("page count returned", doc["page_count"] == 12)
check("chunk count returned", doc["chunk_count"] == 87)
check("size recorded", doc["size_bytes"] == len(PDF))
check("doc id matches vector metadata id",
      doc["id"] == INGESTED[-1]["doc"], f'{doc["id"]} vs {INGESTED[-1]["doc"]}')

print("\n--- streaming ask ---")
with client.stream(
    "POST",
    f"/api/conversations/{conv['id']}/ask/stream",
    json={"question": "What is Gauss's law?"},
) as response:
    check("stream returns 200", response.status_code == 200)
    check("content type is event-stream",
          "text/event-stream" in response.headers.get("content-type", ""))
    body = "".join(response.iter_text())

events = parse_sse(body)
names = [e for e, _ in events]
check("stream opens with start", names[0] == "start")
check("stream ends with done", names[-1] == "done")
check("real stages emitted",
      [n for n in names if n == "stage"].__len__() == 4, str(names))
check("stage order is understanding->searching->reranking->generating",
      [d["stage"] for e, d in events if e == "stage"]
      == ["understanding", "searching", "reranking", "generating"])
check("sources event precedes tokens",
      names.index("sources") < names.index("token"))

tokens = "".join(d["text"] for e, d in events if e == "token")
check("tokens reassemble to full answer", tokens == "".join(ANSWER_TOKENS), tokens)

done = [d for e, d in events if e == "done"][0]
check("done carries message id", bool(done["message_id"]))
check("done reports grounded", done["grounded"] is True)
cited = [c for c in done["citations"] if c["cited"]]
check("only source 1 marked cited", len(cited) == 1 and cited[0]["id"] == 1)
check("citation page from metadata", cited[0]["page"] == 7)
check("citation filename from metadata", cited[0]["source"] == "physics.pdf")

print("\n--- persistence ---")
messages = client.get(f"/api/conversations/{conv['id']}/messages").json()
check("both turns persisted", len(messages) == 2, str([m["role"] for m in messages]))
check("user turn stored verbatim",
      messages[0]["content"] == "What is Gauss's law?")
check("assistant turn stored", messages[1]["content"] == "".join(ANSWER_TOKENS))
check("citations persisted through SQLite",
      messages[1]["citations"][0]["page"] == 7)
check("grounded flag persisted", messages[1]["grounded"] is True)

convs = client.get(f"/api/sessions/{session_id}/conversations").json()
check("conversation auto-titled from first question",
      convs[0]["title"].startswith("What is Gauss"), str(convs[0]["title"]))
check("message count tracked", convs[0]["message_count"] == 2)

print("\n--- memory uses original wording ---")
with client.stream(
    "POST",
    f"/api/conversations/{conv['id']}/ask/stream",
    json={"question": "and the second one?"},
) as response:
    body2 = "".join(response.iter_text())
events2 = parse_sse(body2)
interp = [d for e, d in events2 if e == "interpretation"]
check("follow-up triggered condensation", len(interp) == 1, str(events2[:3]))
messages = client.get(f"/api/conversations/{conv['id']}/messages").json()
check("stored user text is the original, not the rewrite",
      messages[2]["content"] == "and the second one?", messages[2]["content"])
check("interpretation recorded separately",
      messages[3]["interpreted_as"] is not None)

print("\n--- regenerate truncates ---")
assistant_id = messages[3]["id"]
with client.stream(
    "POST",
    f"/api/conversations/{conv['id']}/ask/stream",
    json={"question": "and the second one?", "replace_message_id": assistant_id},
) as response:
    "".join(response.iter_text())
after = client.get(f"/api/conversations/{conv['id']}/messages").json()
check("regenerate did not duplicate the user turn",
      len(after) == 4, str([m["role"] for m in after]))
check("regenerate replaced the assistant turn",
      after[3]["id"] != assistant_id)

print("\n--- session rehydration ---")
state = client.get(f"/api/sessions/{session_id}").json()
check("state returns documents", len(state["documents"]) == 1)
check("state returns conversations", len(state["conversations"]) == 1)

unknown = "11111111-2222-3333-4444-555555555555"
r = client.get(f"/api/sessions/{unknown}")
check("unknown session id adopted, not 404", r.status_code == 200, r.text)
check("adopted session starts empty", r.json()["documents"] == [])

print("\n--- conversation management ---")
second = client.post(f"/api/sessions/{session_id}/conversations").json()
check("second conversation created on same session", second["id"] != conv["id"])
check("documents shared across conversations",
      len(client.get(f"/api/sessions/{session_id}/documents").json()) == 1)

r = client.patch(f"/api/conversations/{second['id']}", json={"title": "Thermo"})
check("rename works", r.status_code == 200 and r.json()["title"] == "Thermo", r.text)

r = client.delete(f"/api/conversations/{second['id']}")
check("delete returns 204", r.status_code == 204)
# Regression guard: a 204 route must declare and return no body. Under
# `from __future__ import annotations` a `-> None` return hint resolves to
# NoneType, which FastAPI <=0.116 infers as a response model and rejects.
check("204 body is empty", r.content == b"", repr(r.content))
check("conversation gone",
      client.get(f"/api/conversations/{second['id']}/messages").status_code == 404)

print("\n--- document deletion ---")
r = client.delete(f"/api/sessions/{session_id}/documents/{doc['id']}")
check("document delete returns 204", r.status_code == 204, r.text)
check("document delete body is empty", r.content == b"", repr(r.content))
check("vector deletion requested", DELETED == [(session_id, doc["id"])], str(DELETED))
check("document removed from listing",
      client.get(f"/api/sessions/{session_id}/documents").json() == [])
r = client.delete(f"/api/sessions/{session_id}/documents/{doc['id']}")
check("deleting twice returns 404", r.status_code == 404)

print("\n--- legacy endpoints still work ---")
legacy_id = client.post("/session").json()["session_id"]
check("legacy /session works", bool(legacy_id))
r = client.post(
    f"/upload?session_id={legacy_id}",
    files={"file": ("old.pdf", PDF, "application/pdf")},
)
check("legacy /upload works", r.status_code == 200, r.text)
check("legacy /upload keeps old response shape",
      set(r.json()) == {"session_id", "source", "chunks_stored"}, str(r.json()))
r = client.post("/ask", json={"session_id": legacy_id, "question": "What is flux?"})
check("legacy /ask works", r.status_code == 200, r.text)
check("legacy /ask keeps old response shape",
      set(r.json()) == {"answer", "citations"}, str(r.json()))
check("legacy /ask returns real citations", r.json()["citations"][0]["page"] == 7)
r = client.post("/ask", json={"session_id": "nope", "question": "x"})
check("legacy /ask 404s on unknown session", r.status_code == 404)

print("\n--- validation ---")
r = client.post(f"/api/conversations/{conv['id']}/ask", json={"question": ""})
check("empty question rejected", r.status_code == 422, r.text)
r = client.post("/api/conversations/does-not-exist/ask", json={"question": "hi"})
check("unknown conversation 404s", r.status_code == 404)

spec = client.get("/openapi.json").json()
for path in ("/api/sessions/{session_id}/documents/{doc_id}", "/api/conversations/{conversation_id}"):
    responses = spec["paths"][path]["delete"]["responses"]
    check(f"OpenAPI 204 declares no body for {path}",
          "content" not in responses["204"], str(responses["204"]))

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    sys.exit(1)
print("ALL API INTEGRATION TESTS PASSED")
