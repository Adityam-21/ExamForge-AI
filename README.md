# ExamForge

> A retrieval-augmented study assistant. Upload your own material, ask questions in natural language, and get answers that are grounded in those documents and cited back to the page.

ExamForge is a conversational AI application built around a multi-strategy RAG pipeline. Answers are generated only from material the user uploads, every citation is derived from the actual retrieved passage rather than from the model's prose, and the interface reports the real retrieval work happening behind each answer.

---

## Live

| Service | Platform | Link |
|---|---|---|
| Frontend | Vercel | [exam-forge-ai.vercel.app](https://exam-forge-ai.vercel.app) |
| Backend API | Northflank | [/health](https://p01--examforge-ai--cqw7qvc7vd22.code.run/health) |
| API reference | FastAPI / OpenAPI | [/docs](https://p01--examforge-ai--cqw7qvc7vd22.code.run/docs) |

---

## What it does

**Retrieval**
- Dense retrieval over sentence-transformer embeddings (`BAAI/bge-small-en-v1.5`)
- BM25 sparse keyword retrieval over the same chunk set
- Hybrid fusion of dense + sparse via `EnsembleRetriever` with reciprocal rank fusion
- Multi-Query expansion — an LLM rewrites the question into several phrasings
- HyDE — an LLM drafts a hypothetical textbook answer and that is embedded instead of the question, closing the vocabulary gap between casual questions and academic prose
- Union deduplication across all strategies
- FlashRank cross-encoder reranking of every surviving candidate
- Per-strategy fault tolerance: a transient LLM failure degrades retrieval quality instead of failing the request

**Grounding and evidence**
- Page-level citations built deterministically from the reranked chunks
- Retrieved passages exposed in the UI with document, page, excerpt and match strength
- Explicit distinction between passages the answer *cited* and other passages *reviewed*
- Answers flagged as unsupported when no valid citation was used or retrieval scored below threshold
- Explicit refusal when the material does not contain the answer

**Conversation**
- LangGraph orchestration: `condense → retrieve → generate`
- Conversation memory via question condensation against recent turns
- Multiple conversations per session, each with its own transcript, sharing one document set
- Full transcript persisted, so a refresh or redeploy does not lose history

**Interface**
- Streamed answers over Server-Sent Events with token-level rendering
- Live activity trace reporting genuine pipeline stages
- Markdown rendering with headings, lists, tables, code blocks and LaTeX
- Copy, regenerate, stop generation, and per-error retry
- Document library with per-document removal
- Light and dark themes, responsive from mobile to desktop

---

## Architecture

```text
┌──────────────────────────── React + Vite (Vercel) ────────────────────────────┐
│  App shell · conversation thread · activity trace · source cards · composer   │
│  useReducer state  ·  SSE client  ·  localStorage session + theme             │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │  REST + Server-Sent Events
                                    ▼
┌──────────────────────────── FastAPI (Northflank) ─────────────────────────────┐
│                                                                               │
│  /api/sessions   /api/…/documents   /api/…/conversations   /api/…/ask/stream  │
│                                    │                                          │
│                    ┌───────────────▼────────────────┐                          │
│                    │      LangGraph StateGraph      │                          │
│                    │  condense → retrieve → generate│                          │
│                    └───────────────┬────────────────┘                          │
│                                    │                                           │
│   ┌────────────────────────────────▼──────────────────────────────────┐        │
│   │                        Retrieval pipeline                          │       │
│   │   Dense ─┐                                                         │       │
│   │   BM25  ─┴─► RRF ensemble ─┐                                       │       │
│   │   Multi-Query ─────────────┼─► dedupe ─► FlashRank rerank ─► top-k │       │
│   │   HyDE ────────────────────┘                                       │       │
│   └────────────────────────────────┬──────────────────────────────────┘        │
│                                    │                                           │
│         ┌──────────────────────────┼──────────────────────────┐                │
│         ▼                          ▼                          ▼                │
│   ┌───────────┐            ┌──────────────┐            ┌───────────┐           │
│   │  ChromaDB │            │    SQLite    │            │  Groq LLM │           │
│   │  vectors  │            │ conversations│            │ inference │           │
│   │ (volume)  │            │  (volume)    │            │           │           │
│   └───────────┘            └──────────────┘            └───────────┘           │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Session model

A **session** owns a Chroma collection (the uploaded documents). A session holds many **conversations**, each with its own transcript. That split is what allows starting a new conversation without re-uploading material.

### How citations stay trustworthy

The generation step does **not** report its own citation metadata. Instead:

1. Reranked chunks are numbered `1..N` and passed to the model with their source and page.
2. The model writes plain markdown, citing with `[1]`, `[2]`.
3. Citation objects are built from the **actual chunks**, and each is flagged with whether the answer really cited it.
4. Markers referring to numbers outside the source list map to nothing and cannot fabricate a source.

Every filename and page number a user sees originates from document metadata written at ingest time. Because the answer is plain markdown rather than a JSON envelope, it can also be streamed token by token.

### Activity trace

Stages shown in the UI (`understanding`, `searching`, `expanding`, `reranking`, `generating`) are emitted by the pipeline itself as it executes. No hidden model reasoning is exposed, requested, or simulated — the labels describe application work.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite 8, plain CSS with design tokens |
| Markdown / math | react-markdown, remark-gfm, remark-math, rehype-katex |
| Icons | lucide-react |
| Frontend state | `useReducer` + context (no external state library) |
| Transport | `fetch` + Server-Sent Events |
| API | FastAPI, Uvicorn |
| Orchestration | LangGraph, LangChain |
| Vector store | ChromaDB (disk-persisted) |
| Embeddings | sentence-transformers, `BAAI/bge-small-en-v1.5` |
| Sparse retrieval | rank_bm25 |
| Reranking | FlashRank cross-encoder |
| LLM inference | Groq (`openai/gpt-oss-120b` generation, `llama-3.1-8b-instant` query transformation) |
| Conversation store | SQLite via stdlib `sqlite3` |
| PDF parsing | pypdf |
| Container | Docker |

---

## Project structure

```text
examforge/
├── app/
│   ├── core/
│   │   ├── config.py            env-driven settings
│   │   └── store.py             SQLite persistence
│   ├── services/
│   │   ├── ingestion.py         PDF parsing, chunking, embedding, chunk cache
│   │   ├── retrieval.py         4 strategies, fusion, reranking, stage callbacks
│   │   └── agent.py             LangGraph graph, deterministic citations, streaming
│   └── api/
│       ├── routes.py            /api resource endpoints + SSE
│       ├── legacy.py            original /session, /upload, /ask shims
│       └── schemas.py           request/response models
├── examforge-frontend/
│   ├── src/
│   │   ├── components/          Composer, MessageList, Message, Markdown,
│   │   │                        SourcePanel, ActivityTrace, Sidebar, EmptyState
│   │   ├── lib/api.js           API client + SSE reader
│   │   ├── state/AppContext.jsx reducer + actions
│   │   ├── styles/              tokens.css, app.css
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── .env.example
│   ├── vercel.json
│   └── vite.config.js
├── tests/test_api_integration.py
├── main.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## API

Resource endpoints are under `/api`. The original flat endpoints still work and return their original shapes.

### Sessions
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/sessions` | Create a session |
| `GET` | `/api/sessions/{id}` | Documents + conversations, for client rehydration |

### Documents
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/sessions/{id}/documents` | List documents |
| `POST` | `/api/sessions/{id}/documents` | Upload and ingest a PDF (multipart) |
| `DELETE` | `/api/sessions/{id}/documents/{docId}` | Remove one document and its vectors |

### Conversations
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/sessions/{id}/conversations` | List conversations |
| `POST` | `/api/sessions/{id}/conversations` | Start a conversation |
| `PATCH` | `/api/conversations/{id}` | Rename |
| `DELETE` | `/api/conversations/{id}` | Delete |
| `GET` | `/api/conversations/{id}/messages` | Full transcript with citations |

### Ask
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/conversations/{id}/ask/stream` | SSE: stages, sources, tokens |
| `POST` | `/api/conversations/{id}/ask` | Non-streaming fallback |

Both accept `{ "question": "...", "replace_message_id": null }`. Supplying `replace_message_id` regenerates: that message and everything after it is discarded and the preceding user turn is reused.

### Stream events

| Event | Payload |
|---|---|
| `start` | `{ conversation_id }` |
| `stage` | `{ stage, label }` |
| `interpretation` | `{ question }` — the condensed form of a follow-up |
| `sources` | `{ sources: [...] }` — retrieved passages, before generation |
| `token` | `{ text }` |
| `done` | `{ message_id, grounded, citations, created_at }` |
| `error` | `{ message }` |

### Legacy
`POST /session`, `POST /upload?session_id=`, `POST /ask` — retained, marked deprecated in OpenAPI, implemented on the same pipeline.

---

## Getting started

**Prerequisites:** Python 3.11+, Node 20+, a Groq API key.

### Backend

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
uvicorn main:app --reload
```

API at `http://localhost:8000`, interactive docs at `/docs`.

First start downloads the embedding model (~130 MB). Reranker weights download on the first question.

### Frontend

```bash
cd examforge-frontend
npm install
cp .env.example .env        # VITE_API_URL=http://localhost:8000
npm run dev
```

### Docker

```bash
docker compose up --build
```

Brings up the API on `:8000` and the Vite dev server on `:5173`, with a named volume for vectors and conversation history.

### Tests

```bash
python tests/test_api_integration.py
```

Exercises the real app, routers and store with only the embedding and LLM boundaries stubbed: upload validation, SSE framing and stage ordering, citation integrity, persistence, memory fidelity, regenerate truncation, session rehydration, document deletion, and legacy endpoint compatibility.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** LLM inference |
| `DATA_DIR` | `data` | Root for vectors + database |
| `CHROMA_PATH` | `data/chroma` | Vector store location |
| `DATABASE_PATH` | `data/examforge.db` | Conversation store |
| `CORS_ORIGINS` | localhost + Vercel | Comma-separated allowed origins |
| `CORS_ORIGIN_REGEX` | Vercel previews | Regex for preview deployments |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embeddings |
| `UTILITY_MODEL` | `llama-3.1-8b-instant` | Query transformation |
| `GENERATION_MODEL` | `openai/gpt-oss-120b` | Answer generation |
| `RETRIEVER_K` | `5` | Per-strategy retrieval depth |
| `RERANK_TOP_N` | `5` | Chunks kept after reranking |
| `MIN_RELEVANCE_SCORE` | `0.05` | Below this, answers are flagged unsupported |
| `MEMORY_WINDOW` | `6` | Turns fed to question condensation |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `50` | Chunking |
| `MAX_UPLOAD_MB` | `10` | Upload limit |

Frontend: `VITE_API_URL` only.

---

## Deployment

**Backend (Northflank).** Build from the `Dockerfile`. Mount a persistent volume at `/app/data` so vectors and conversation history survive redeploys. Set `GROQ_API_KEY` and `CORS_ORIGINS`. Run one worker — the embedding model and reranker are in-process and the stores are local files.

**Frontend (Vercel).** Root directory `examforge-frontend`, framework preset Vite. Set `VITE_API_URL` to the backend URL. `vercel.json` handles SPA rewrites.

---

## Engineering notes

Decisions and trade-offs worth calling out:

- **Citations are derived, not declared.** Generation previously returned a JSON envelope containing its own citation metadata, which meant page numbers could be invented and a single unparseable response returned HTTP 500. Numbering the chunks and parsing markers back to them removed both the trust problem and the parsing fragility, and made token streaming possible.
- **SQLite over in-memory dicts.** Conversation state previously lived in module-level dicts, so a redeploy 404'd every active session while leaving orphaned Chroma collections on disk. Stdlib `sqlite3` on the same volume as the vector store fixed persistence without adding a dependency or a service.
- **Streaming stages are real.** Progress events are emitted by the pipeline as it runs, not simulated on a timer. This matters because the pipeline genuinely takes time: several LLM round trips plus reranking.
- **BM25 index is cached per session.** Previously every question dumped the whole collection from Chroma and rebuilt the index. The cache is invalidated whenever a session's documents change.
- **Retrieval strategies fail independently.** A Multi-Query or HyDE timeout degrades recall instead of failing the request.
- **Legacy endpoints retained.** The published API contract still works, implemented over the same pipeline rather than duplicated.

---

## Not implemented

Stated plainly, because none of this is in the codebase:

- **Authentication.** There are no user accounts. A session id is an unguessable UUID, and anyone holding one can access that session. Fine for a demo; not a multi-tenant security model.
- **Metadata filtering.** Metadata is attached and displayed but never used to constrain retrieval.
- **RAG evaluation.** No offline evaluation harness or retrieval quality metrics.
- **Non-PDF ingestion.** PDF only. Scanned/image-only PDFs are rejected with a clear message rather than OCR'd.
- **Frontend test suite.** The SSE client and store logic are covered by the integration tests; there are no component tests.
- **Horizontal scale.** Single-worker by design (in-process models, local file stores). Scaling out would need a shared vector service and a networked database.
- **Monitoring.** Structured logging and a health endpoint only; no metrics or tracing backend.

---

## Author

**Kumar Adityam** — [GitHub](https://github.com/Adityam-21) · [LinkedIn](https://www.linkedin.com/in/kumar-adityam)

## License

MIT.
