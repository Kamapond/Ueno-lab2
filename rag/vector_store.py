"""
vector_store.py — chunking and vector-store construction
(paper: Appendix B; repository module `rag/`).

The offline phase of the RAG pipeline: the knowledge-source documents are split into retrieval
units and written to a persistent vector store, so that indexing is paid for once and reused
across every question and every model.

Chunking. Splitting is recursive over a hierarchy of separators, which keeps natural semantic
units of Japanese text intact rather than cutting at a fixed offset. A chunk of 500 characters
with an overlap of 150 (30 %) was adopted after a pilot comparison: the overlap prevents a
requirement sentence from being severed at a chunk boundary and preserves semantic continuity
between neighbouring chunks. Each chunk carries its ordinal position, so the surrounding
context can be reconstructed when a retrieved passage is presented.

Deterministic identity. A chunk identifier is the SHA-256 digest of its text together with its
source and page. The same chunk therefore always receives the same identifier, which makes
ingestion idempotent: re-running it adds only genuinely new chunks and never duplicates
existing ones, so documents can be added incrementally without rebuilding the store.

Batching. Embeddings are requested in small batches so that large corpora can be indexed
without exceeding the embedding endpoint's per-request token and rate limits.
"""

from __future__ import annotations

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Adopted after the pilot comparison (paper: Section III-E-2).
CHUNK_SIZE = 500        # characters
CHUNK_OVERLAP = 150     # characters (30 %)
EMBED_BATCH_SIZE = 64

# Hierarchical separators: paragraph, line, Japanese sentence end, clause, space, character.
SEPARATORS = ["\n\n", "\n", "。", "、", " ", ""]


def split_docs(docs):
    """Split documents into retrieval units, tagging each with its ordinal position."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator=False,
    )
    chunks = splitter.split_documents(docs)

    for i, d in enumerate(chunks):
        d.metadata["chunk_index"] = i
    return chunks


def deterministic_id(text: str, meta: dict) -> str:
    """Content-addressed chunk identifier, so that re-ingestion is idempotent."""
    base = (
        f"{meta.get('source')}|{meta.get('page')}|{meta.get('chunk_index')}|{text}"
    ).encode("utf-8")
    return hashlib.sha256(base).hexdigest()


def ingest_chunked(chunks, vector_store, existing_ids=None) -> int:
    """Add only chunks not already present. Returns the number newly added.

    Args:
        chunks:       output of `split_docs`.
        vector_store: any store exposing `add_documents(documents, ids=...)`.
        existing_ids: identifiers already in the store; queried from the store when omitted.
    """
    if existing_ids is None:
        existing_ids = get_all_ids(vector_store)

    new_docs, new_ids = [], []
    for d in chunks:
        did = deterministic_id(d.page_content, d.metadata)
        if did not in existing_ids:
            new_docs.append(d)
            new_ids.append(did)

    if new_docs:
        for start in range(0, len(new_docs), EMBED_BATCH_SIZE):
            stop = start + EMBED_BATCH_SIZE
            vector_store.add_documents(new_docs[start:stop], ids=new_ids[start:stop])

    return len(new_docs)


def get_all_ids(vector_store) -> set[str]:
    """Identifiers already held by the store.

    Left to the deployment: the call differs between vector-store backends and between
    versions of the same backend. For a Chroma collection this is `collection.get()["ids"]`.
    """
    raise NotImplementedError("Provide the deployment's vector-store identifier query.")
