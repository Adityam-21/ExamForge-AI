from typing import TypedDict, Annotated
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from app.services.retrieval import retrieve
import json

from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

retrieval_llm = ChatGroq(
    model="openai/gpt-oss-20b", temperature=0, api_key=GROQ_API_KEY
)

generation_llm = ChatGroq(
    model="openai/gpt-oss-120b", temperature=0.3, api_key=GROQ_API_KEY
)


class ExamForgeState(TypedDict):
    session_id: str
    question: str
    memory: list[dict]
    context: list[Document]
    answer: str
    citations: list[dict]


def memory_node(state: ExamForgeState) -> ExamForgeState:
    memory = state["memory"]
    question = state["question"]

    if not memory:
        return state

    memory_str = "\n".join(
        [f"{msg['role'].upper()}: {msg['content']}" for msg in memory[-5:]]
    )

    summary_prompt = f"""Given this conversation history:
    {memory_str}

    And this new question: {question}

    Rewrite the question to be self contained, incorporating any relevant context from the history.
    Return only the rewritten question, nothing else."""

    rewritten = generation_llm.invoke(summary_prompt).content

    return {**state, "question": rewritten}


def retrieval_node(state: ExamForgeState) -> ExamForgeState:
    chunks = retrieve(state["session_id"], state["question"])
    return {**state, "context": chunks}


def generation_node(state: ExamForgeState) -> ExamForgeState:

    question = state["question"]
    chunks = state["context"]
    memory = state["memory"]

    context_str = "\n\n".join(
        [
            f"[Source: {doc.metadata['source']}, Page {doc.metadata['page']}] \n {doc.page_content}"
            for doc in chunks
        ]
    )

    prompt = f"""You are an academic assistant helping a student understand their study material.
    Answer the question using ONLY the provided context below.
    Always cite your sources using [Source: filename, Page X] format.

    For broad questions like "what are the main concepts" or "what might come in the exam":
    - Identify and list all key themes and concepts from the context
    - Be comprehensive, don't miss anything important
    - Organise your answer clearly with headings if needed

    For specific questions:
    - Answer precisely and concisely
    - Cite the exact source

    If the answer is not in the context, say "I couldn't find this in your uploaded material."

    CONTEXT:
    {context_str}

    QUESTION:
    {question}

    Respond ONLY in this JSON format, no markdown, no preamble:
    {{
       "answer": "your answer here with citation markers like [1], [2]",
       "citations": [
            {{
              "id": 1,
              "source": "filename.pdf",
              "page": 4,
              "chunk": "exact chunk text this citation refers to"
            }}
        ]
    }}

    ANSWER:"""

    response = generation_llm.invoke(prompt).content
    parsed = json.loads(response)
    answer = parsed["answer"]
    citations = parsed["citations"]

    updated_memory = memory + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    updated_memory = updated_memory[-10:]

    return {**state, "answer": answer, "citations": citations, "memory": updated_memory}


def build_graph():
    graph = StateGraph(ExamForgeState)

    graph.add_node("memory_processor", memory_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("generation", generation_node)

    graph.set_entry_point("memory_processor")

    graph.add_edge("memory_processor", "retrieval")
    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", END)

    return graph.compile()


examforge_graph = build_graph()
