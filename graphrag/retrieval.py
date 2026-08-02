"""
retrieval.py — two-axis scoring and Skyline selection
(paper: Appendix C; repository module `graphrag/`).

Where plain RAG retrieves points — the passages nearest the question in embedding space —
GraphRAG reasons along lines: it enters the graph at the specialist terms named by the question
and follows MENTIONS edges to every passage that discusses them. Retrieval is therefore scored
on two axes that measure different things:

    Sg  graph score — the number of distinct schema entities a chunk matches. Structural
        relevance: how densely the passage participates in the terminology of the question.
    St  text score  — cosine similarity between the question and the chunk embedding.
        Semantic relevance: how closely the passage reads like an answer.

Selection is Pareto-optimal rather than a weighted sum. A chunk is kept when no other candidate
is at least as good on both axes and strictly better on one. This deliberately avoids choosing a
trade-off weight between structure and semantics, which would have to be tuned per corpus; the
Skyline keeps every candidate that is defensible on its own terms, so a passage that is
structurally central survives even if its wording diverges from the question, and vice versa.

Keyword extraction. Search terms are extracted from the question by the same model that
produces the answer, rather than the question being sent to the graph verbatim: the query is an
examination item, and its incidental wording would otherwise dilute the entity match.

NOTE ON THE PUBLISHED SOURCE. The abridged listing in the appendix of the underlying Japanese
manuscript shows `limit=5`, an artefact of the abridgement. The configuration used throughout
the reported experiments selects 50 chunks from a candidate pool ten times that size, as stated
in Section III-E-3.
"""

from __future__ import annotations

import numpy as np

# Retrieval configuration (paper: Section III-E-3)
SKYLINE_LIMIT = 50           # chunks handed to the generator
CANDIDATE_POOL_FACTOR = 10   # candidate pool = SKYLINE_LIMIT * this

# Graph search: entry at the matched entities, then out along MENTIONS to the passages, and up
# to the owning section so the answer can name where the evidence sits.
CONTEXT_QUERY = """
UNWIND $terms AS term
MATCH (e:Entity) WHERE e.id CONTAINS term
MATCH (c:Chunk)-[:MENTIONS]->(e)
MATCH (s:Section)-[:HAS_CHILD]->(c)
RETURN c.text AS text,
       s.title AS section,
       c.page AS page,
       count(DISTINCT e) AS score_g
ORDER BY score_g DESC
LIMIT $pool
"""


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two embedding vectors."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def skyline_filter(candidates: list[dict], limit: int = SKYLINE_LIMIT) -> list[dict]:
    """Keep the Pareto-optimal candidates on (Sg, St).

    A candidate is dominated when another is at least as good on both axes and strictly better
    on at least one. Both the surviving frontier and the dominated remainder are then ordered by
    summed score, so that truncation to `limit` keeps the strongest of the frontier rather than
    an arbitrary subset, and a frontier smaller than `limit` is topped up from the best of the
    remainder rather than leaving the context starved.
    """
    if not candidates:
        return []

    skyline, others = [], []

    for i, candidate_a in enumerate(candidates):
        dominated = False
        for j, candidate_b in enumerate(candidates):
            if i == j:
                continue
            if (candidate_b["score_g"] >= candidate_a["score_g"]
                    and candidate_b["score_t"] >= candidate_a["score_t"]
                    and (candidate_b["score_g"] > candidate_a["score_g"]
                         or candidate_b["score_t"] > candidate_a["score_t"])):
                dominated = True
                break

        (skyline if not dominated else others).append(candidate_a)

    skyline.sort(key=lambda x: x["score_g"] + x["score_t"], reverse=True)
    others.sort(key=lambda x: x["score_g"] + x["score_t"], reverse=True)

    if len(skyline) < limit:
        return skyline + others[:limit - len(skyline)]

    return skyline[:limit]


class GraphRetriever:
    """Retrieve context for a question by graph traversal, then Skyline selection."""

    def __init__(self, driver, embedding_model, llm):
        self.driver = driver
        self.embedding_model = embedding_model
        self.llm = llm

    def get_context(self, query: str, limit: int = SKYLINE_LIMIT) -> list[dict]:
        """Return the selected chunks, each with its Sg and St scores attached."""
        search_terms = self.extract_search_entities(query)

        with self.driver.session() as session:
            records = list(session.run(
                CONTEXT_QUERY,
                terms=search_terms,
                pool=limit * CANDIDATE_POOL_FACTOR,
            ))

        query_vec = self.embedding_model.embed_query(query)

        candidates = []
        for record in records:
            chunk_vec = self.embedding_model.embed_documents([record["text"]])[0]
            candidates.append({
                "data": dict(record),
                "score_g": record["score_g"],
                "score_t": cosine_similarity(query_vec, chunk_vec),
                "matched_keywords": search_terms,
            })

        return skyline_filter(candidates, limit)

    def extract_search_entities(self, query: str) -> list[str]:
        """Specialist terms to enter the graph with, extracted from the question by the LLM.

        Left to the deployment: the extraction instruction is written in the language of the
        benchmark and is bound to the calling convention of the client in use.
        """
        raise NotImplementedError("Provide the deployment's keyword-extraction call.")
