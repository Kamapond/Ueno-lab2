"""
chain.py — LCEL retrieve-and-generate chain
(paper: Appendix B, Fig. 14; repository module `rag/`).

The online phase of the RAG pipeline, expressed declaratively with LangChain Expression
Language. The question is passed straight through to the prompt, the retriever fetches the
related passages, `format_docs` renders them with their provenance, and the whole assembly
behaves as a single runnable invoked synchronously per item.

Retrieval depth. The retriever returns the five passages most similar to the question. Five
covers the typical span of a specialist reference — several clauses and their definitions —
while keeping the context dense; the prompt, question and context together come to roughly
5,000 tokens, comfortably inside the input window of every model tested.

Provenance. Each passage is rendered with its source file and page so that the answer can cite
them, which is what makes a response auditable rather than merely plausible.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .prompts import RESPONSE_PROMPT

TOP_K = 5


def format_docs(docs) -> str:
    """Render retrieved passages with their provenance, one block per passage."""
    blocks = []
    for d in docs:
        source = d.metadata.get("source")
        page = d.metadata.get("page")
        blocks.append(f"[Source: {source} | p.{page}]\n{d.page_content}\n")
    return "\n".join(blocks)


def build_retriever(vector_store, k: int = TOP_K):
    """Similarity-search interface over the vector store."""
    return vector_store.as_retriever(search_kwargs={"k": k})


def build_chain(vector_store, llm, k: int = TOP_K):
    """Assemble the retrieve-and-generate chain.

    Returns a runnable taking the question string and returning the model's answer text.
    """
    retriever = build_retriever(vector_store, k)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RESPONSE_PROMPT
        | llm
        | StrOutputParser()
    )
