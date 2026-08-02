"""
extraction.py — Step 1 of knowledge-graph construction: extraction
(paper: Appendix C, Fig. 15; repository module `graphrag/`).

Construction is split into extraction and ingestion so that the schema can be adjusted, or the
data refreshed, without rebuilding the other half. This module covers extraction: documents are
parsed into chunks, entities and relations are extracted from each chunk under the domain
schema, and the result is written as intermediate tables.

Parsing is heading-aware, which is the substantive difference from the fixed-length splitting
used for the plain RAG condition (`rag/vector_store.py`). Splitting on the heading hierarchy
keeps a clause together with its own heading, so the document's own structure survives into the
graph as the tree that later carries traversal.

Extraction runs one request per chunk at temperature 0, so that the mapping from chunk to
extracted triples is deterministic and reproducible.

Intermediate tables. Extraction writes four tables — chunks, entities, mentions and
relationships — before anything reaches the graph database. Materializing them makes the
extraction auditable and correctable prior to ingestion, and the mentions table records which
chunk mentions which entity, which is the basis of the GT-Link built during ingestion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from .schema import ALLOWED_NODES, ALLOWED_RELATIONSHIPS

# Extraction settings (paper: Appendix C-B)
EXTRACTION_MODEL = "gpt-5-mini"
EXTRACTION_TEMPERATURE = 0
CHUNKS_PER_REQUEST = 1

# Intermediate table names written before ingestion.
INTERMEDIATE_TABLES = ("chunks", "entities", "mentions", "relationships")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class ChunkData:
    """One retrieval unit, carrying the position it occupied in the source document."""

    chunk_id: str
    doc_id: str
    text: str
    section: str = ""
    page: int | None = None
    hierarchy: list[str] = field(default_factory=list)
    is_table: bool = False


class MarkdownParser:
    """Split a document into chunks along its heading hierarchy.

    Follows the parsing behaviour described in Appendix C: headings establish the hierarchy,
    pipe-delimited runs are kept together as tables rather than split mid-row, and each chunk
    records its document, page, section title and full heading path.
    """

    def parse_file(self, file_path: Path, doc_id: str, page: int | None = None) -> list[ChunkData]:
        markdown = Path(file_path).read_text(encoding="utf-8")
        return self.parse_text(markdown, doc_id=doc_id, page=page)

    def parse_text(self, markdown: str, doc_id: str, page: int | None = None) -> list[ChunkData]:
        chunks: list[ChunkData] = []
        hierarchy: list[str] = []
        buffer: list[str] = []
        in_table = False

        def flush() -> None:
            nonlocal buffer, in_table
            text = "\n".join(buffer).strip()
            if text:
                chunks.append(ChunkData(
                    chunk_id=f"{doc_id}:{len(chunks)}",
                    doc_id=doc_id,
                    text=text,
                    section=hierarchy[-1] if hierarchy else "",
                    page=page,
                    hierarchy=list(hierarchy),
                    is_table=in_table,
                ))
            buffer, in_table = [], False

        for line in markdown.splitlines():
            heading = HEADING_RE.match(line)
            if heading:
                flush()
                level, title = len(heading.group(1)), heading.group(2).strip()
                del hierarchy[level - 1:]
                hierarchy.append(title)
                continue

            is_row = bool(TABLE_ROW_RE.match(line))
            if is_row and not in_table:
                flush()          # a table starts its own chunk
                in_table = True
            elif in_table and not is_row:
                flush()          # and ends with the last row
            buffer.append(line)

        flush()
        return chunks


class GraphExtractor:
    """Extract entities and relations from each chunk under the domain schema."""

    def __init__(self, model_name: str = EXTRACTION_MODEL):
        self.llm = ChatOpenAI(temperature=EXTRACTION_TEMPERATURE, model=model_name)

        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=ALLOWED_NODES,                    # 25 types (TABLE VII)
            allowed_relationships=ALLOWED_RELATIONSHIPS,    # 103 types (TABLE VIII)
        )

    def extract(self, documents):
        """Return the extracted graph documents, one request per chunk."""
        results = []
        for start in range(0, len(documents), CHUNKS_PER_REQUEST):
            batch = documents[start:start + CHUNKS_PER_REQUEST]
            results.extend(self.transformer.convert_to_graph_documents(batch))
        return results
