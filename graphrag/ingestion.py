"""
ingestion.py — Step 2 of knowledge-graph construction: ingestion
(paper: Appendix C, Fig. 15; repository module `graphrag/`).

The intermediate tables produced by `extraction.py` are imported into Neo4j AuraDB, a native
graph database that represents and traverses a node-and-edge structure directly. Ingestion runs
in five stages, and their order matters:

  1. Constraints and indexes — uniqueness on the chunk, entity and document identifiers, and
     search indexes on the section title and entity type. Created first, so that the MERGE
     operations that follow are both deduplicating and fast.
  2. Skeleton — the Document to Section to Chunk hierarchy as HAS_CHILD, plus NEXT between
     consecutive chunks of a document, so that the paragraph before or after a retrieved chunk
     can be reached in one hop.
  3. Labels — each entity receives a dynamic label from its type attribute, which makes
     type-filtered search efficient.
  4. Relations — the semantic relations between entities, so that specialist statements such as
     "material A requires condition B" or "instrument C detects defect D" exist on the graph.
  5. GT-Link — MENTIONS between chunks and entities. This is the join between the tree (the
     document's structure) and the graph (the network of knowledge), and it is what makes the
     traversal bidirectional: from a term to every passage that mentions it, and from a passage
     to every term it contains.

The resulting graph holds 62,983 nodes and 174,429 relations.

Deliberate abstractions in this published version
  * The database URI and credentials are read from the environment; none are embedded.
  * Writes are executed inside per-session transactions, batched by the caller.
"""

from __future__ import annotations

import os

from neo4j import GraphDatabase

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — constraints and indexes
# ─────────────────────────────────────────────────────────────────────────────
CONSTRAINTS = [
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX section_title IF NOT EXISTS FOR (s:Section) ON (s.title)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
]

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — document skeleton (tree)
# ─────────────────────────────────────────────────────────────────────────────
TREE_QUERY = """
MERGE (d:Document {id: $docId})
MERGE (s:Section {title: $section, docId: $docId})
MERGE (d)-[:HAS_CHILD]->(s)
MERGE (c:Chunk {id: $chunkId})
MERGE (s)-[:HAS_CHILD]->(c)
"""

NEXT_QUERY = """
MATCH (c1:Chunk {id: $curr_id})
MATCH (c2:Chunk {id: $next_id})
MERGE (c1)-[:NEXT]->(c2)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — GT-Link
# ─────────────────────────────────────────────────────────────────────────────
MENTIONS_QUERY = """
MATCH (c:Chunk {id: $chunkId})
MATCH (e:Entity {id: $entityId})
MERGE (c)-[:MENTIONS]->(e)
"""


def driver_from_env():
    """Neo4j driver built from NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD."""
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )


class GraphIngestor:
    """Import the intermediate tables into the graph database."""

    def __init__(self, driver=None):
        self.driver = driver or driver_from_env()

    def close(self) -> None:
        self.driver.close()

    def create_constraints(self) -> None:
        """Stage 1."""
        with self.driver.session() as session:
            for statement in CONSTRAINTS + INDEXES:
                session.run(statement)

    def import_chunks_and_tree(self, df_chunks) -> None:
        """Stage 2 — Document/Section/Chunk hierarchy, then chunk ordering within a document."""
        with self.driver.session() as session:
            for row in df_chunks.itertuples():
                session.run(
                    TREE_QUERY,
                    docId=row.doc_id,
                    section=row.section,
                    chunkId=row.chunk_id,
                )

            for doc_id, group in df_chunks.groupby("doc_id"):
                ids = list(group["chunk_id"])
                for curr_id, next_id in zip(ids, ids[1:]):
                    session.run(NEXT_QUERY, curr_id=curr_id, next_id=next_id)

    def import_entities(self, df_entities) -> None:
        """Stage 3 — entities with a dynamic label taken from the type attribute."""
        with self.driver.session() as session:
            for row in df_entities.itertuples():
                session.run(
                    "MERGE (e:Entity {id: $entityId}) "
                    "SET e.type = $type, e.name = $name "
                    "WITH e CALL apoc.create.addLabels(e, [$type]) YIELD node RETURN node",
                    entityId=row.entity_id,
                    type=row.type,
                    name=row.name,
                )

    def import_relationships(self, df_relationships) -> None:
        """Stage 4 — semantic relations between entities.

        The relation type is validated against the schema before it is interpolated into the
        statement, because Cypher does not parameterize relationship types.
        """
        from .schema import ALLOWED_RELATIONSHIPS

        allowed = set(ALLOWED_RELATIONSHIPS)
        with self.driver.session() as session:
            for row in df_relationships.itertuples():
                rel_type = row.type
                if rel_type not in allowed:
                    continue
                session.run(
                    "MATCH (a:Entity {id: $sourceId}) "
                    "MATCH (b:Entity {id: $targetId}) "
                    f"MERGE (a)-[:{rel_type}]->(b)",
                    sourceId=row.source_id,
                    targetId=row.target_id,
                )

    def import_mentions(self, df_mentions) -> None:
        """Stage 5 — the GT-Link joining the tree to the graph."""
        with self.driver.session() as session:
            for row in df_mentions.itertuples():
                session.run(
                    MENTIONS_QUERY,
                    chunkId=row.chunk_id,
                    entityId=row.entity_id,
                )
