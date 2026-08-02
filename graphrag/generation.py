"""
generation.py — GraphRAG answer generation
(paper: Appendix C, TABLE VI; repository module `graphrag/`).

The selected chunks are rendered with their provenance and both retrieval scores, then passed
through the same grounding prompt used for the plain RAG condition (`rag/prompts.py`). Holding
the prompt constant across the two retrieval conditions is deliberate: it means any difference
in measured ability is attributable to how context is selected, not to how it is presented.

Annotating each passage with its own Sg and St, and with the keywords it matched, does more
than aid debugging — it makes the retrieval decision legible in the answer itself, so a
reviewer can see why a given passage was placed before the model. Combined with the
`Context_Unspecified` marker, this is what makes a GraphRAG answer auditable: which evidence
was supplied, on what grounds, and whether the answer actually rested on it.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from rag.prompts import RESPONSE_PROMPT


def format_context(selected: list[dict]) -> str:
    """Render the selected chunks with provenance and retrieval scores.

    Fields follow TABLE VI: source, section, page, both scores, and the matched keywords.
    """
    blocks = []
    for cand in selected:
        record = cand["data"]
        blocks.append(
            f"[Source: Neo4j | Section: {record['section']} | p.{record.get('page')}]\n"
            f"[Scores: Sg={cand['score_g']}, St={cand['score_t']:.3f}]\n"
            f"[Matched Keywords: {', '.join(cand.get('matched_keywords', []))}]\n"
            f"{record['text']}\n"
        )
    return "\n".join(blocks)


def answer(question: str, retriever, llm) -> str:
    """Retrieve context by graph traversal and Skyline selection, then generate the answer."""
    selected = retriever.get_context(question)
    context_str = format_context(selected)

    chain = RESPONSE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context_str})
