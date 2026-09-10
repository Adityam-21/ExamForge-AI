# D1 — LangSmith tracing

## 1. Dependency

Add to `requirements.txt` (it was removed earlier for exactly the right reason — nothing used it; now something does):

```
# --- Observability -----------------------------------------------------------
langsmith==0.1.147
```

## 2. Environment

Local `.env` and Northflank secrets. **Never committed** — `.env` is already gitignored.

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=examforge-prod        # examforge-ci in CI, examforge-dev locally
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

Separate projects per environment. Mixing CI runs into the production project makes latency percentiles meaningless.

## 3. What leaves your infrastructure — read this before enabling in production

LangSmith is third-party hosted. With tracing on, these are transmitted and stored on LangSmith's servers:

| Data | Sent? | Sensitivity |
|---|---|---|
| The user's question, verbatim | **Yes** | User content |
| The condensed/rewritten question | **Yes** | Derived user content |
| Multi-Query rewrites and the HyDE hypothetical | **Yes** | Derived |
| **Retrieved chunk text from uploaded PDFs** | **Yes** | **Highest — this is the user's own study material** |
| The generated answer | **Yes** | Derived |
| Latency, token counts, model names, tags | Yes | Operational, harmless |
| Uploaded PDF files themselves | No | — |
| Session ids | Yes, if tagged | Pseudonymous |

The row that matters is the retrieved chunks. Someone uploads a textbook, a set of lecture notes, possibly something they don't own the rights to — and the passages land in a third-party SaaS.

Three options, pick consciously:

1. **Trace everything.** Fine for a portfolio project you control. Say so in the README.
2. **Trace metadata only in production, full traces in dev.** Set `LANGCHAIN_HIDE_INPUTS=true` / `LANGCHAIN_HIDE_OUTPUTS=true` on the production deployment, keep full traces locally and in CI where the corpus is your own eval PDFs. **This is what I'd recommend** — you keep latency and token data, which is what D4 needs, and user material stays home.
3. **Dev/CI only.** Simplest, but you lose production latency data, which is the number actually going on your CV.

Recommendation: option 2. It also gives you a genuinely good interview answer about privacy trade-offs in observability.

## 4. Where the trace boundaries go

LangChain components (`ChatGroq`, `EnsembleRetriever`, `MultiQueryRetriever`, the LangGraph nodes) instrument themselves. You only need `@traceable` on the plain-Python steps that would otherwise be invisible.

**`app/api/routes.py`** — the SSE handler. This is the trace root; everything else nests under it.

```python
from langsmith import traceable

@traceable(
    name="examforge_ask",
    run_type="chain",
    tags=["production"],
    metadata={
        "generation_model": config.GENERATION_MODEL,
        "utility_model": config.UTILITY_MODEL,
        "retriever_k": config.RETRIEVER_K,
        "rerank_top_n": config.RERANK_TOP_N,
        "chunk_size": config.CHUNK_SIZE,
        "rerank_enabled": True,
    },
)
def _traced_pipeline(session_id: str, question: str, memory: list, emit):
    return run_pipeline(session_id, question, memory, emit=emit)
```

Call `_traced_pipeline` from the route instead of `run_pipeline`. The route runs the graph in a worker thread — LangSmith's context is thread-local, so the decorator must wrap the function that *runs inside the worker*, not the async handler.

**`app/services/retrieval.py`** — the fusion layer is plain Python and is exactly where you want visibility.

```python
@traceable(run_type="retriever", name="retrieve")
def retrieve(session_id, question, on_stage=None): ...

@traceable(run_type="retriever", name="dense_and_sparse")
def _dense_and_sparse(...): ...

@traceable(run_type="retriever", name="hyde")
def _hyde(...): ...

@traceable(run_type="retriever", name="rerank")
def rerank(question, chunks): ...
```

`_multi_query` wraps `MultiQueryRetriever`, which traces itself — decorating it too just adds a redundant span. Skip it.

`_safe` should **not** be decorated. It wraps every strategy; decorating it produces four identically-named spans and makes the tree unreadable.

**`app/services/agent.py`** — `generation_node` already contains a `ChatGroq.stream()` call, which traces automatically with token counts. Don't decorate the node.

### Tagging

Untagged runs cannot be compared later. The `metadata` dict on the root is what lets you filter "all runs with rerank_top_n=5 and gpt-oss-120b" three weeks from now. This is the single highest-value five minutes in D1 — set it up properly the first time.

## 5. Done when

You can open one request in LangSmith and see a tree like:

```
examforge_ask                    2.9s
├─ condense_node                 0.4s   142 in / 18 out
├─ retrieve                      1.6s
│  ├─ dense_and_sparse           0.2s
│  ├─ MultiQueryRetriever        0.7s   88 in / 61 out
│  ├─ hyde                       0.6s   54 in / 130 out
│  └─ rerank                     0.1s
└─ generation_node               0.9s   2,840 in / 310 out
```

Paste that tree back. The per-step token counts in the right column are what D4 turns into a cost-per-query figure.
