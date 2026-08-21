# Architecture & Decision Record

Companion to the README. The README describes *what* the system does; this describes *why* it is built this way, and what was deliberately left out.

---

## 1. Starting position

An audit of the pre-existing codebase found a backend substantially more capable than its documentation or its frontend suggested:

- Hybrid dense + BM25 retrieval with reciprocal rank fusion — **implemented**, documented as a future improvement
- FlashRank cross-encoder reranking — **implemented**, documented as a future improvement
- Conversation memory via question condensation — **implemented**, documented as a future improvement
- Page-level citations returned by the API — **implemented**, documented as a future improvement
- Multi-Query and HyDE retrieval — **implemented**, documented nowhere
- LangGraph orchestration — **implemented**, documented nowhere

Meanwhile the frontend read `data.answer` from the ask response and discarded `data.citations` entirely. A user could not perceive any of the retrieval engineering.

The transformation was therefore framed as an **exposure problem first and a correctness problem second** — not a rewrite. The retrieval pipeline was preserved strategy-for-strategy.

---

## 2. Decisions

### 2.1 Citations derived from chunks, not declared by the model

**Problem.** Generation returned a JSON envelope in which the model reported its own citation metadata:

```json
{ "answer": "...", "citations": [{ "id": 1, "source": "notes.pdf", "page": 4, "chunk": "..." }] }
```

Two failures followed. A page number could be invented, because nothing tied it to a retrieved chunk. And `json.loads` was unguarded, so any fenced or prefaced response raised and returned HTTP 500 — the most likely cause of intermittent failures.

**Decision.** Number the reranked chunks `1..N`, pass them to the model with source and page, and have the model write **plain markdown** citing `[1]`, `[2]`. Build citation objects from the actual chunks and flag each with whether the answer cited it.

```python
cited = {int(n) for n in CITATION_PATTERN.findall(answer)}
sources = [{ "id": i, "source": md.get("source"), "page": md.get("page"),
             "text": chunk.page_content, "relevance_score": ...,
             "cited": i in cited } for i, chunk in enumerate(chunks, 1)]
```

**Consequences.**
- Every filename and page originates from ingest metadata. A marker like `[7]` against three sources maps to nothing.
- The JSON parsing failure mode is gone entirely.
- The answer is now streamable token by token, which the JSON envelope had made impossible.
- The UI can distinguish *cited* evidence from *reviewed* evidence, which is a more honest representation of what RAG actually did.

**Trade-off.** The model could still cite a valid source that doesn't genuinely support its sentence. Attribution is verifiable; entailment is not checked. `MIN_RELEVANCE_SCORE` and the `grounded` flag mitigate but do not solve this. Fixing it properly needs a claim-level verification pass, which was out of scope.

### 2.2 SQLite for conversation state

**Problem.** `memory_store: dict[str, list]` and `uploaded_sessions: set[str]` were module-level Python objects. A redeploy 404'd every active session while its Chroma collection stayed on disk, orphaned forever with no cleanup. Multiple workers would route a session to the wrong process.

**Options.**
| Option | Verdict |
|---|---|
| Keep in-memory | Rejected — the failure is user-visible on every deploy |
| JSON file | Rejected — no concurrency story, full rewrite per turn |
| Postgres | Rejected — a service and a dependency for four small tables |
| **stdlib `sqlite3`** | **Chosen** |

**Rationale.** Zero new dependencies. A single file on the volume already mounted for Chroma, so vectors and history persist or fail together. WAL mode plus a connection per operation is safe under FastAPI's threadpool and adequate for this workload.

**Consequence.** Sessions, documents, conversations, messages and citations all persist. It also unlocked history and document-listing endpoints from data that already existed.

### 2.3 Session owns documents; conversations are nested

The original model conflated three things into one `session_id`: the Chroma collection, the document set, and the transcript. "New conversation" would therefore have meant re-uploading.

```
session  ──owns──►  documents  (one Chroma collection)
   └─────has many──►  conversations  ──has many──►  messages
```

Retrieval spans all documents in the session; memory is scoped to the conversation. Small schema change, and it is what makes the sidebar coherent.

### 2.4 Streaming stages are emitted, not simulated

The pipeline makes several sequential LLM round trips plus reranking, so the wait is genuinely long. A spinner would waste that time, and a timer-driven fake would be dishonest.

An `emit` callable is threaded through the graph's runtime config. Nodes and the retrieval pipeline call it as they progress. The route runs the sync graph in a worker thread and pumps events into an `asyncio.Queue` via `loop.call_soon_threadsafe`, which the SSE generator drains.

```
worker thread                    event loop
  condense_node ──emit──► queue ──► SSE ──► client
  retrieve()    ──emit──►
  generate()    ──emit──►  (token by token)
```

**Constraint honoured.** Stage labels describe *application work* — `searching`, `reranking`, `generating`. No private model reasoning is exposed, requested, or invented. The one interpretive signal shown, `interpretation`, is the condensed question, which is a real artefact the pipeline produces and uses for retrieval.

**Stop generation.** The client aborts the fetch. The generator's `finally` block persists whatever tokens were produced, so the stored transcript matches what the user actually saw.

### 2.5 Frontend state: reducer over library

One `useReducer` in `AppContext` owns bootstrap, documents, conversations, transcript and live stream. Components are presentational.

Rejected: Redux/Zustand (a single reducer suffices at this size), React Query (one SSE flow and a handful of fetches), a component library (would fight the "not templated" goal).

Accepted dependencies, each with a reason: `react-markdown` + `remark-gfm` (the generation prompt explicitly requests headings, lists and tables), `remark-math` + `rehype-katex` (study material is often STEM), `lucide-react` (tree-shaken; better than hand-rolling twenty SVGs).

`STREAM_COMMIT` converts the finished stream into a transcript entry in a single update. Clearing the stream and waiting for the server reload blanked the answer for a frame.

### 2.6 Performance: cache the BM25 index

Every question previously dumped the entire collection from Chroma and rebuilt the BM25 index from scratch. Chunks are now cached per session in `ingestion.get_all_chunks`, invalidated whenever a session's documents change.

Remaining cost is inherent: condensation, Multi-Query, HyDE and generation are four sequential LLM calls. Reducing that would mean dropping a retrieval strategy — a quality regression, and explicitly not the goal.

### 2.7 Fault isolation in retrieval

Each strategy runs through a `_safe` wrapper. A Multi-Query or HyDE timeout degrades recall and logs a warning rather than failing the request. Reranking falls back to fusion order. Previously any single failure returned a 500.

### 2.8 Legacy endpoints retained

`POST /session`, `POST /upload`, `POST /ask` are preserved with their original response shapes, marked deprecated in OpenAPI, and implemented over the new pipeline. The published API contract does not break, and there is one code path rather than two.

---

## 3. Bugs fixed

| Bug | Detail |
|---|---|
| Temp filenames in citations | `load_pdf` used `Path(file_path).name` — the `NamedTemporaryFile` path — so every citation read `tmpq8x_31ab.pdf`. `original_filename` was accepted but only used in the return value. |
| Unguarded JSON parse | Removed by design change (§2.1). |
| Unsafe metadata access | `doc.metadata['source']` → `.get()` with defaults. |
| Memory stored rewritten questions | Transcript now stores the user's original wording; the condensed form is kept separately as `interpreted_as`. |
| Session loss on redeploy | §2.2, plus unknown ids are adopted rather than 404'd. |
| Session loss on refresh | Session id in `localStorage`, rehydrated via `GET /api/sessions/{id}`. |
| Hardcoded API URL | `VITE_API_URL`, with `.env.example` and `vercel.json`. |
| Dead model reference | `llama-3.1-70b-versatile` (decommissioned by Groq) client removed; unused imports dropped. |
| Import-time crash without a key | LLM clients built lazily behind `lru_cache`. |
| Broken compose frontend service | Referenced a non-existent Dockerfile and passed `NEXT_PUBLIC_*` to a Vite app. Replaced with a working dev-server service. |
| Phantom `data.message` | Upload UI read a field the API never returned; now shows real page and chunk counts. |
| Unused dependencies | `ragas`, `reportlab`, `langsmith` removed — pinned but with no supporting code. |

---

## 4. Testing

`tests/test_api_integration.py` runs the real app, routers and store; only embedding/vector storage and the Groq client are stubbed. Coverage:

- Upload validation — wrong type, empty, oversized, and the original filename reaching ingest
- SSE — framing, stage ordering, `sources` before tokens, token reassembly, terminal `done`
- Citation integrity — only genuinely cited sources flagged; page and filename from metadata
- Persistence — both turns, citations round-tripped through SQLite, `grounded` flag, auto-titling
- Memory fidelity — stored user text is the original, not the rewrite
- Regenerate — truncates without duplicating the user turn
- Rehydration — unknown session adopted, not rejected
- Document deletion — vectors requested, listing updated, second delete 404s
- Legacy endpoints — original response shapes preserved

Separately verified during development: the SSE client parser against frames split mid-JSON, multibyte characters split across chunk boundaries, malformed frames, unterminated tails and abort propagation; and the citation/grounding functions against out-of-range markers, refusals, sub-threshold retrieval and missing metadata.

Not covered: component rendering tests, and end-to-end tests against live Groq and a real PDF. Both were traded for delivery time; the first is the more valuable addition.

---

## 5. Deliberately out of scope

Each of these was considered and rejected for this delivery:

- **Authentication.** Session ids are unguessable UUIDs but confer full access. Correct for a demo, insufficient for multi-tenancy. Adding it would have consumed the window and changed the product shape.
- **Metadata filtering.** Metadata is attached and displayed; using it to constrain retrieval needs a UI for expressing filters, which is a feature, not a fix.
- **RAG evaluation.** Genuinely valuable and the most defensible next addition, but it is an offline harness — invisible in the product and a poor use of a two-day window.
- **Cloud vector storage.** Would remove the single-worker constraint. Not needed at current scale.
- **DOCX/TXT ingestion.** Breadth, not depth.
- **TypeScript migration.** Real long-term value, no user-visible benefit, high regression risk under time pressure.

---

## 6. If continuing

In priority order:

1. **RAG evaluation harness** — a small labelled question set with retrieval hit-rate and faithfulness scoring. Would let retrieval changes be justified with numbers instead of intuition, and is the single strongest signal for an AI/ML reviewer.
2. **Claim-level grounding check** — a verification pass over cited sentences, closing the entailment gap in §2.1.
3. **Component tests** — the composer, source panel and markdown renderer.
4. **Retrieval latency** — run Multi-Query and HyDE concurrently rather than sequentially; roughly a third off the wait for no quality cost.
5. **Auth + shared vector service** — only if this becomes multi-user.
