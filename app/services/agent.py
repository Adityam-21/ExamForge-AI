"""LangGraph orchestration for a grounded answer.

Graph: condense -> retrieve -> generate

Three deliberate changes from the previous implementation:

1. **Citations are deterministic.** Previously the model was asked to emit a JSON
   object containing its own citation metadata, so a page number could simply be
   invented, and an unparseable response raised and returned HTTP 500. Now the
   reranked chunks are numbered and handed to the model, the model writes plain
   markdown with ``[n]`` markers, and the citation objects are built from the
   *actual* chunks. Every page number the user sees comes from the retrieved
   document, never from the model.

2. **The answer is plain markdown**, which means it can be streamed token by
   token and rendered progressively. Structured JSON output made streaming
   impossible.

3. **Memory stores the user's original question**, not the condensed rewrite, so
   conversation history matches what the user actually typed.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Callable, TypedDict

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.core import config
from app.services.retrieval import relevance_of, retrieve

logger = logging.getLogger(__name__)

NO_EVIDENCE_MESSAGE = (
    "I couldn't find anything about this in your uploaded material."
)

@lru_cache(maxsize=1)
def get_generation_llm() -> ChatGroq:
    return ChatGroq(
        model=config.GENERATION_MODEL,
        temperature=0.3,
        api_key=config.GROQ_API_KEY,
    )


@lru_cache(maxsize=1)
def get_condense_llm() -> ChatGroq:
    return ChatGroq(
        model=config.UTILITY_MODEL,
        temperature=0,
        api_key=config.GROQ_API_KEY,
    )

CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")


class ExamForgeState(TypedDict, total=False):
    session_id: str
    question: str          # what the user actually typed
    search_query: str      # condensed, self-contained form used for retrieval
    memory: list[dict[str, str]]
    context: list[Document]
    answer: str
    citations: list[dict[str, Any]]
    grounded: bool


# --- Event emitter -----------------------------------------------------------
# Streaming is driven by an emitter passed through the graph's runtime config.
# Nodes report real progress; nothing here fabricates model reasoning.

Emitter = Callable[[str, dict[str, Any]], None]


def _emitter(cfg: dict | None) -> Emitter:
    if not cfg:
        return lambda event, payload: None
    emit = (cfg.get("configurable") or {}).get("emit")
    return emit or (lambda event, payload: None)


# --- Nodes -------------------------------------------------------------------

CONDENSE_PROMPT = """Given this conversation history:
{history}

And this new question: {question}

Rewrite the question to be self-contained, incorporating any relevant context
from the history. Return only the rewritten question, nothing else."""


def condense_node(state: ExamForgeState, config_: dict | None = None) -> ExamForgeState:
    """Resolve follow-up questions against conversation history."""
    emit = _emitter(config_)
    question = state["question"]
    memory = state.get("memory") or []

    emit("stage", {"stage": "understanding", "label": "Understanding your question"})

    if not memory:
        return {**state, "search_query": question}

    history = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in memory[-config.MEMORY_WINDOW:]
    )

    try:
        rewritten = get_condense_llm().invoke(
            CONDENSE_PROMPT.format(history=history, question=question)
        ).content.strip()
    except Exception:  # noqa: BLE001
        logger.warning("Question condensation failed; using original", exc_info=True)
        return {**state, "search_query": question}

    if not rewritten:
        rewritten = question

    # Only report the rewrite when it materially differs - otherwise it is noise.
    if rewritten.lower().strip(" ?.") != question.lower().strip(" ?."):
        emit("interpretation", {"question": rewritten})

    return {**state, "search_query": rewritten}


def retrieval_node(state: ExamForgeState, config_: dict | None = None) -> ExamForgeState:
    emit = _emitter(config_)

    def on_stage(stage: str, label: str) -> None:
        emit("stage", {"stage": stage, "label": label})

    chunks = retrieve(state["session_id"], state.get("search_query") or state["question"], on_stage)

    sources = build_sources(chunks)
    emit("sources", {"sources": sources})

    return {**state, "context": chunks}


ANSWER_PROMPT = """You are ExamForge, an academic assistant helping a student \
understand their own study material.

Answer the question using ONLY the numbered sources below.

Citation rules:
- Cite with bracketed numbers that match the source numbers, e.g. [1] or [2][3].
- Place each citation immediately after the claim it supports.
- Never cite a number that is not in the source list.
- Never invent page numbers or filenames in your prose; the citation markers
  carry that information.

Answering rules:
- For broad questions ("what are the main concepts", "what might come in the
  exam"), identify every key theme present in the sources, be comprehensive, and
  organise the answer with markdown headings and lists.
- For specific questions, answer precisely and concisely.
- Use markdown: headings, bullet lists, tables, bold for key terms, and LaTeX
  between $...$ or $$...$$ for mathematics.
- If the sources do not contain the answer, reply with exactly:
  "{no_evidence}"
- Never pad an answer with general knowledge that is not in the sources.

SOURCES:
{context}

QUESTION:
{question}

ANSWER:"""


def generation_node(state: ExamForgeState, config_: dict | None = None) -> ExamForgeState:
    emit = _emitter(config_)
    chunks = state.get("context") or []
    question = state.get("search_query") or state["question"]

    if not chunks:
        emit("stage", {"stage": "generating", "label": "Generating answer"})
        emit("token", {"text": NO_EVIDENCE_MESSAGE})
        return {
            **state,
            "answer": NO_EVIDENCE_MESSAGE,
            "citations": [],
            "grounded": False,
        }

    context_str = "\n\n".join(
        f"[{i}] {chunk.metadata.get('source', 'document')} "
        f"· page {chunk.metadata.get('page', '?')}\n{chunk.page_content}"
        for i, chunk in enumerate(chunks, start=1)
    )

    prompt = ANSWER_PROMPT.format(
        no_evidence=NO_EVIDENCE_MESSAGE,
        context=context_str,
        question=question,
    )

    emit("stage", {"stage": "generating", "label": "Generating answer"})

    answer = ""
    try:
        for part in get_generation_llm().stream(prompt):
            text = getattr(part, "content", "") or ""
            if text:
                answer += text
                emit("token", {"text": text})
    except Exception:  # noqa: BLE001
        logger.exception("Generation failed")
        if not answer:
            raise

    answer = answer.strip()
    if not answer:
        answer = NO_EVIDENCE_MESSAGE

    citations = build_sources(chunks, answer)
    grounded = is_grounded(answer, chunks, citations)

    return {**state, "answer": answer, "citations": citations, "grounded": grounded}


# --- Citation construction ---------------------------------------------------


def build_sources(
    chunks: list[Document], answer: str | None = None
) -> list[dict[str, Any]]:
    """Build citation objects from the real reranked chunks.

    Every field originates from document metadata written at ingest time, so a
    page number can never be hallucinated. When ``answer`` is supplied, each
    source is additionally flagged with whether the model actually cited it,
    letting the UI foreground cited evidence while still exposing everything
    that was retrieved.
    """
    cited: set[int] = set()
    if answer:
        cited = {int(n) for n in CITATION_PATTERN.findall(answer)}

    sources = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata or {}
        sources.append(
            {
                "id": index,
                "doc_id": metadata.get("doc_id"),
                "source": metadata.get("source", "document"),
                "page": metadata.get("page"),
                "chunk_index": metadata.get("chunk_index"),
                "text": chunk.page_content,
                "relevance_score": round(relevance_of(chunk), 4),
                "cited": index in cited if answer else False,
            }
        )
    return sources


def is_grounded(
    answer: str, chunks: list[Document], citations: list[dict[str, Any]]
) -> bool:
    """Whether the answer is genuinely supported by retrieved evidence."""
    if not chunks:
        return False
    if answer.strip().startswith(NO_EVIDENCE_MESSAGE[:40]):
        return False
    if not any(c["cited"] for c in citations):
        return False
    best = max((relevance_of(c) for c in chunks), default=0.0)
    return best >= config.MIN_RELEVANCE_SCORE


# --- Graph -------------------------------------------------------------------


def build_graph():
    graph = StateGraph(ExamForgeState)

    graph.add_node("condense", condense_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("generate", generation_node)

    graph.set_entry_point("condense")
    graph.add_edge("condense", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


examforge_graph = build_graph()


def run_pipeline(
    session_id: str,
    question: str,
    memory: list[dict[str, str]],
    emit: Emitter | None = None,
) -> ExamForgeState:
    """Invoke the graph. ``emit`` receives real progress events."""
    return examforge_graph.invoke(
        {
            "session_id": session_id,
            "question": question,
            "search_query": question,
            "memory": memory,
            "context": [],
            "answer": "",
            "citations": [],
            "grounded": False,
        },
        config={"configurable": {"emit": emit or (lambda e, p: None)}},
    )
