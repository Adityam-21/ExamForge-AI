"""Tests for the retrieval-grounded abstention guard in generation_node.

The abstention decision is deterministic given the reranked chunks' scores, so
these tests stub only the LLM boundary and drive generation_node directly with
controlled relevance_score values. This verifies the gate fires BEFORE the LLM
is called, not merely that the answer is labelled afterwards.

Run:  python tests/test_abstention.py
"""

from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("DATA_DIR", "/tmp/examforge-abstain")
os.environ.setdefault("GROQ_API_KEY", "test-key")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Stub the ML boundary ----------------------------------------------------

class _Doc:
    def __init__(self, page_content="", metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __or__(self, other):
        return self

    def invoke(self, *a, **k):
        return self


_mod("langchain_core")
_mod("langchain_core.documents", Document=_Doc)
_mod("langchain_core.prompts", PromptTemplate=_Stub)
_mod("langchain_groq", ChatGroq=_Stub)


class _SG(_Stub):
    def add_node(self, *a, **k): pass
    def set_entry_point(self, *a): pass
    def add_edge(self, *a): pass
    def compile(self): return object()


_mod("langgraph")
_mod("langgraph.graph", StateGraph=_SG, END="END")
_mod("langchain")
_mod("langchain.retrievers", EnsembleRetriever=_Stub, ContextualCompressionRetriever=_Stub)
_mod("langchain.retrievers.multi_query", MultiQueryRetriever=_Stub)
_mod("langchain_community")
_mod("langchain_community.document_compressors", FlashrankRerank=_Stub)
_mod("langchain_community.retrievers", BM25Retriever=_Stub)
_mod("langchain_community.vectorstores", Chroma=_Stub)
_mod("langchain_community.embeddings", HuggingFaceEmbeddings=lambda **k: None)
_mod("langchain_community.document_loaders", PyPDFLoader=_Stub)
_mod("langchain_text_splitters", RecursiveCharacterTextSplitter=_Stub)

from app.core import config  # noqa: E402
from app.services import agent  # noqa: E402


# --- Controllable fake generation LLM ---------------------------------------

class _FakeStreamPart:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """Streams a fixed answer and records whether it was called at all."""

    called = False

    def stream(self, prompt):
        _FakeLLM.called = True
        # A confident answer that cites [1] - the kind the gate must prevent
        # from ever being produced on weak retrieval.
        for token in ["This ", "is ", "a confident answer [1]."]:
            yield _FakeStreamPart(token)


def _install_fake_llm():
    _FakeLLM.called = False
    agent.get_generation_llm = lambda: _FakeLLM()


def _chunk(text, source, page, score):
    return _Doc(
        text,
        {
            "source": source,
            "page": page,
            "chunk_index": 0,
            "doc_id": "d1",
            "relevance_score": score,
        },
    )


def _collect_emits():
    events = []
    cfg = {"configurable": {"emit": lambda e, p: events.append((e, p))}}
    return events, cfg


# --- Harness -----------------------------------------------------------------

FAILURES = []


def check(name, cond, extra=""):
    print(("  ok    " if cond else "  FAIL  ") + name + ("" if cond else f"  {extra}"))
    if not cond:
        FAILURES.append(name)


THRESHOLD = config.MIN_RELEVANCE_SCORE
print(f"\nMIN_RELEVANCE_SCORE = {THRESHOLD}\n")


# --- 1. Clearly relevant -> normal grounded answer ---------------------------

print("--- relevant question -> generates ---")
_install_fake_llm()
strong = [
    _chunk("Gauss's law relates flux to charge.", "physics.pdf", 7, 0.88),
    _chunk("Faraday's law concerns induced EMF.", "physics.pdf", 9, 0.41),
]
events, cfg = _collect_emits()
result = agent.generation_node(
    {"question": "What is Gauss's law?", "search_query": "What is Gauss's law?",
     "context": strong},
    cfg,
)
check("LLM was called for strong evidence", _FakeLLM.called)
check("answer is the generated text", result["answer"].startswith("This is a confident"))
check("grounded is True", result["grounded"] is True)
check("citations present", len(result["citations"]) == 2)
check("cited source flagged", any(c["cited"] for c in result["citations"]))


# --- 2. Clearly irrelevant -> abstention -------------------------------------

print("\n--- irrelevant question -> abstains, LLM never called ---")
_install_fake_llm()
weak = [
    _chunk("Appendix table of constants.", "physics.pdf", 1, 0.004),
    _chunk("Index of terms.", "physics.pdf", 30, 0.001),
]
events, cfg = _collect_emits()
result = agent.generation_node(
    {"question": "What is the capital of France?",
     "search_query": "What is the capital of France?", "context": weak},
    cfg,
)
check("LLM was NOT called on weak evidence", _FakeLLM.called is False)
check("answer is the abstention message",
      result["answer"] == agent.NO_EVIDENCE_MESSAGE)
check("grounded is False when abstaining", result["grounded"] is False)
check("citations empty when abstaining", result["citations"] == [])


# --- 3. Borderline (exactly at threshold) -> still generates -----------------

print("\n--- borderline (== threshold) -> still generates (no hard refusal) ---")
_install_fake_llm()
borderline = [
    _chunk("Tangentially related passage.", "physics.pdf", 3, THRESHOLD),
    _chunk("Weaker passage.", "physics.pdf", 4, THRESHOLD / 2),
]
events, cfg = _collect_emits()
result = agent.generation_node(
    {"question": "borderline question", "search_query": "borderline question",
     "context": borderline},
    cfg,
)
check("LLM called at exactly threshold (borderline preserved)", _FakeLLM.called)
check("borderline answer is generated, not abstained",
      result["answer"] != agent.NO_EVIDENCE_MESSAGE)


# --- 3b. Just below threshold -> abstains ------------------------------------

print("\n--- just below threshold -> abstains ---")
_install_fake_llm()
just_below = [_chunk("Barely related.", "physics.pdf", 5, THRESHOLD - 0.001)]
events, cfg = _collect_emits()
result = agent.generation_node(
    {"question": "q", "search_query": "q", "context": just_below}, cfg
)
check("LLM NOT called just below threshold", _FakeLLM.called is False)
check("abstains just below threshold", result["answer"] == agent.NO_EVIDENCE_MESSAGE)


# --- 4. Empty retrieval -> abstains (pre-existing path still works) ----------

print("\n--- empty retrieval -> abstains ---")
_install_fake_llm()
events, cfg = _collect_emits()
result = agent.generation_node({"question": "q", "search_query": "q", "context": []}, cfg)
check("LLM NOT called on empty retrieval", _FakeLLM.called is False)
check("empty retrieval abstains", result["answer"] == agent.NO_EVIDENCE_MESSAGE)
check("empty retrieval grounded False", result["grounded"] is False)
check("empty retrieval citations empty", result["citations"] == [])


# --- 5. SSE semantics on the abstention path ---------------------------------

print("\n--- abstention still emits valid stage + token events ---")
_install_fake_llm()
events, cfg = _collect_emits()
agent.generation_node(
    {"question": "q", "search_query": "q", "context": weak}, cfg
)
names = [e for e, _ in events]
check("emits a 'stage' event", "stage" in names)
check("stage is 'generating'",
      any(p.get("stage") == "generating" for e, p in events if e == "stage"))
check("emits a 'token' event with the abstention text",
      any(e == "token" and p.get("text") == agent.NO_EVIDENCE_MESSAGE
          for e, p in events))


# --- 6. Node sequence: retrieval_node -> generation_node keeps SSE contract ---
# Drives the real nodes in order (the compiled graph object is stubbed out in
# this test harness, so we exercise the same node code the graph would run).

print("\n--- retrieval->generation abstention keeps SSE contract ---")

# Stub retrieve() so retrieval_node returns weak chunks deterministically.
agent.retrieve = lambda session_id, q, on_stage=None: (
    [on_stage("searching", "Searching")] and weak if on_stage else weak
)
_install_fake_llm()

captured = []
cfg = {"configurable": {"emit": lambda e, p: captured.append((e, p))}}

state = {
    "session_id": "s1",
    "question": "unrelated question",
    "search_query": "unrelated question",
    "memory": [],
    "context": [],
    "answer": "",
    "citations": [],
    "grounded": False,
}
state = agent.retrieval_node(state, cfg)
state = agent.generation_node(state, cfg)

ev = [e for e, _ in captured]
check("node flow abstains", state["answer"] == agent.NO_EVIDENCE_MESSAGE)
check("node flow grounded False", state["grounded"] is False)
check("node flow citations empty", state["citations"] == [])
check("pipeline emitted stage events", "stage" in ev)
check("pipeline emitted token event", "token" in ev)
check("LLM never called through node flow", _FakeLLM.called is False)


print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    sys.exit(1)
print("ALL ABSTENTION TESTS PASSED")
