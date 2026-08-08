from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_groq import ChatGroq
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from app.services.ingestion import embedding_model, CHROMA_PATH
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

retrieval_llm = ChatGroq(
    model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_API_KEY
)

generation_llm = ChatGroq(
    model="llama-3.1-70b-versatile", temperature=0.3, api_key=GROQ_API_KEY
)


def get_chunks_from_chroma(session_id: str) -> list[Document]:
    vectorstore = Chroma(
        collection_name=session_id,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH,
    )

    results = vectorstore.get()

    chunks = []
    for i, doc in enumerate(results["documents"]):
        chunks.append(Document(page_content=doc, metadata=results["metadatas"][i]))

    return chunks


def get_multi_query_retriever(session_id: str) -> MultiQueryRetriever:

    vectorstore = Chroma(
        collection_name=session_id,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH,
    )

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=retrieval_llm
    )

    return multi_query_retriever


def get_hyde_retriever(session_id: str) -> ContextualCompressionRetriever:

    vectorstore = Chroma(
        collection_name=session_id,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH,
    )

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    hyde_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Generate a hypothetical answer to the following question.
        This answer will be used to search a document database, not shown to the user.
        Write it as if it came directly from a textbook or academic document.
        Be concise, factual, and specific.

        Question: {question}
        Hypothetical Answer:""",
    )

    hyde_chain = hyde_prompt | retrieval_llm

    class HyDERetriever:
        def __init__(self, chain, retriever):
            self.chain = chain
            self.retriever = retriever

        def invoke(self, question: str) -> list[Document]:
            fake_answer = self.chain.invoke({"question": question}).content
            return self.retriever.invoke(fake_answer)

    return HyDERetriever(hyde_chain, base_retriever)


def get_ensemble_retriever(session_id: str, chunks: list[Document]):

    vectorstore = Chroma(
        collection_name=session_id,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH,
    )

    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5], c=60
    )

    return ensemble_retriever


def deduplicate(chunks: list[Document]) -> list[Document]:

    seen = set()
    unique = []

    for chunk in chunks:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            unique.append(chunk)

    return unique


def rerank(question: str, chunks: list[Document]) -> list[Document]:

    if not chunks:
        return []

    compressor = FlashrankRerank(top_n=5)

    reranked = compressor.compress_documents(
        documents=chunks,
        query=question,
    )

    return reranked


def retrieve(session_id: str, question: str) -> list[Document]:

    all_chunks = get_chunks_from_chroma(session_id)

    if not all_chunks:
        raise ValueError("No document content is available for this session.")

    multi_query_retriever = get_multi_query_retriever(session_id)
    hyde_retriever = get_hyde_retriever(session_id)
    ensemble_retriever = get_ensemble_retriever(session_id, all_chunks)

    multi_query_chunks = multi_query_retriever.invoke(question)
    ensemble_chunks = ensemble_retriever.invoke(question)
    hyde_chunks = hyde_retriever.invoke(question)

    all_retrieved = multi_query_chunks + ensemble_chunks + hyde_chunks

    unique_chunks = deduplicate(all_retrieved)

    if not unique_chunks:
        return []

    final_chunks = rerank(question, unique_chunks)

    return final_chunks
