"""
draft_jis_clauses.py — reference implementation of the ISO-to-JIS clause-drafting pipeline
(paper: Section VI; repository module `jis_drafting/`).

One infrastructure, two tasks: the domain knowledge graph (KG) and the BookRAG hybrid
retrieval validated on closed-form question answering (repository modules `rag/` and `graphrag/`) are reused
here for the open-form task of drafting national-standard clauses from international
source standards.

Pipeline (identifiers give the corresponding part of the paper)
    Sec. VI-A  segment_source_clauses()   source standard -> one record per clause (heading split)
    Sec. VI-A  load_terminology_library() committee-approved renderings, indexed per clause
    Sec. VI-B  retrieve_evidence()        tiered corpus -> hybrid (Sg, St) scoring -> Skyline filter
    Sec. VI-B  build_drafting_request()   category-constrained generation prompt
    Sec. VI-B  verify_color_key()         color-key symmetry check + one regeneration attempt
    Sec. VI-B  linkify_citations()        chunk identifiers -> resolvable source-passage links

Scoring convention (inverted relative to question answering, cf. Appendix C-D): the source
clause is English and the corpus Japanese, so the text score St (cosine similarity) is the
primary axis and the graph score Sg (matched specialist entities) is auxiliary — it rewards
terms written identically in both languages, such as the method acronym itself.

Deliberate abstractions in this published version
  * Credentials and endpoints are read from the environment; none are embedded.
  * Deployment-specific paths, embedding caches, and the document-assembly stage (Section VI-C) are
    replaced by the injected `Corpus` / `SourceResolver` protocols.
  * The normative Japanese content of the generation constraints — the frozen quantitative
    requirement values, the JIS Z 8301:2019 modal-force map, and the terminology table — is
    loaded from external configuration and is NOT reproduced here: those values and terms
    are drawn from copyrighted standards. `prompts/drafting_system.md` and `config/*.yaml`
    in this repository hold the machine-readable schemas with the normative payload removed.

Usage
    python draft_jis_clauses.py --standard ISO13588 --clause 6.1 7.3.1
    python draft_jis_clauses.py --standard ISO19285 --new-only
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Tiered corpus (paper: Section VI-B, TABLE IV)
#
# Each tier holds a fixed governance role, and a per-tier quota keeps any single
# tier from crowding the evidence set. Document prefixes identify the standard,
# never its text.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Tier:
    id: str                    # P1..P7
    role: str                  # governance role in the draft
    prefixes: tuple[str, ...]  # source-document identifier prefixes
    quota: int                 # chunks retained after Skyline filtering


TIERS: tuple[Tier, ...] = (
    Tier("P1", "terminology",             ("Z2300", "NDIS2002"),                     5),
    Tier("P2", "existing procedures",     ("UT_Z3060", "UT_Z3070", "UT_Z2344"),      8),
    Tier("P3", "PAUT-specific basis",     ("NDIS2429",),                             5),
    Tier("P4", "material grounding",      ("UT_G", "UT_Z306", "UT_Z308", "UT_H"),    5),
    Tier("P5", "instrument performance",  ("UT_Z234", "UT_Z235"),                    5),
    Tier("P6", "technical principles",    ("JSNDI_lv2", "JSNDI_lv3"),                8),
    Tier("P7", "personnel qualification", ("QUALIFICATION",),                        3),
)

# Skyline operates on the top `quota * CANDIDATE_MULTIPLIER` chunks of each tier.
CANDIDATE_MULTIPLIER = 10

# Entities shorter than this produce spurious substring matches in Sg.
KEYWORD_MIN_LEN = 3

# Text budgets for the generation prompt.
EXCERPT_CHARS = 350   # evidence excerpt per chunk
SOURCE_CHARS = 4000   # source-clause text

# Clause categories (paper: Section VI-B). The category fixes what the generator may adapt and
# what it must carry across unchanged.
CATEGORIES = (
    "management-framework",
    "technical-procedure",
    "quantitative-value-freeze",
    "equipment-and-specimen",
    "PAUT-new",
)


# ─────────────────────────────────────────────────────────────────────────────
# Injected dependencies — keep deployment detail out of the algorithm
# ─────────────────────────────────────────────────────────────────────────────

class Corpus(Protocol):
    """Read access to the knowledge graph and its embedded chunks."""

    def clause_entities(self, standard: str, clause: str) -> list[dict]:
        """Specialist entities (and their values) mentioned by one source clause.

        Cypher, walking the Document->Section->Chunk tree and the MENTIONS edges
        built in Appendix C-C:

            MATCH (d:Document {id: $standard})-[:HAS_CHILD*]->(s:Section)
            WHERE s.title STARTS WITH $clause_prefix
            MATCH (s)-[:HAS_CHILD*]->(c:Chunk)-[:MENTIONS]->(e:Entity)
            OPTIONAL MATCH (e)-[:HAS_VALUE]->(v:Entity)
            RETURN e.id AS entity, v.id AS value, c.page AS page
        """

    def chunks_by_id(self, chunk_ids: Sequence[str]) -> list[dict]:
        """Retrieval seed: the cached top existing-JIS passages for a clause (Section VI-A)."""

    def chunks_with_prefix(self, prefixes: Sequence[str]) -> list[dict]:
        """Candidate pool for one tier: dicts of id, text, page, embedding."""

    def clause_embedding(self, standard: str, clause: str) -> np.ndarray | None:
        """Embedding of the source clause; None falls back to St = 0."""


class SourceResolver(Protocol):
    """Maps a chunk identifier to a stable, citable location of its source passage."""

    def uri(self, chunk_id: str) -> str: ...


class Generator(Protocol):
    """LLM wrapper. `system` carries the constraints of Section VI-B."""

    def complete(self, system: str, user: str) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retrieval and Skyline selection (paper: Section VI-B)
# ─────────────────────────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def skyline_filter(candidates: list[dict], limit: int) -> list[dict]:
    """Retain the Pareto-optimal chunks on (Sg, St), then top up by summed score.

    Candidate b dominates a when b is at least equal on both axes and strictly
    better on one; dominated candidates are demoted rather than discarded, so the
    quota is always met.
    """
    if not candidates:
        return []
    skyline, dominated = [], []
    for a in candidates:
        is_dominated = any(
            b is not a
            and b["Sg"] >= a["Sg"] and b["St"] >= a["St"]
            and (b["Sg"] > a["Sg"] or b["St"] > a["St"])
            for b in candidates
        )
        (dominated if is_dominated else skyline).append(a)
    key = lambda c: c["Sg"] + c["St"]
    skyline.sort(key=key, reverse=True)
    dominated.sort(key=key, reverse=True)
    return (skyline + dominated)[:limit] if len(skyline) < limit else skyline[:limit]


def score_tier(
    tier: Tier,
    keywords: Sequence[str],
    clause_vec: np.ndarray | None,
    corpus: Corpus,
) -> list[dict]:
    """Score one tier on both axes and return its Skyline-selected evidence."""
    pool = corpus.chunks_with_prefix(tier.prefixes)
    if not pool:
        return []

    terms = [k.lower() for k in keywords if len(k) >= KEYWORD_MIN_LEN]
    candidates = []
    for chunk in pool:
        text = chunk.get("text") or ""
        lowered = text.lower()
        emb = chunk.get("embedding")
        candidates.append({
            "citation_id": chunk["id"],
            "text": text[:EXCERPT_CHARS],
            "page": chunk.get("page"),
            # Sg: auxiliary axis — number of distinct matched specialist entities.
            "Sg": sum(1 for t in terms if t in lowered),
            # St: primary axis — cross-lingual semantic proximity to the source clause.
            "St": cosine(clause_vec, emb) if clause_vec is not None and emb is not None else 0.0,
            "tier": tier.id,
        })

    candidates.sort(key=lambda c: c["Sg"] + c["St"], reverse=True)
    return skyline_filter(candidates[: tier.quota * CANDIDATE_MULTIPLIER], tier.quota)


def retrieve_evidence(
    standard: str,
    clause: str,
    seed_chunk_ids: Sequence[str],
    corpus: Corpus,
    title: str = "",
) -> dict:
    """Assemble the per-clause evidence set: entities, retrieval seed, tiered chunks."""
    entities = corpus.clause_entities(standard, clause)
    keywords = sorted({
        e["entity"].strip() for e in entities
        if e.get("entity") and len(e["entity"].strip()) >= KEYWORD_MIN_LEN
    })
    # Entity-sparse clauses fall back to content words of the clause title.
    if len(keywords) < 3 and title:
        keywords = sorted(set(keywords) | set(re.findall(r"[A-Za-z]{4,}", title)))

    clause_vec = corpus.clause_embedding(standard, clause)
    return {
        "entities": entities,
        "keywords": keywords,
        "seed": corpus.chunks_by_id(seed_chunk_ids),
        "tiers": {t.id: score_tier(t, keywords, clause_vec, corpus) for t in TIERS},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Source segmentation and terminology library (paper: Section VI-A)
# ─────────────────────────────────────────────────────────────────────────────

HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)
CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+")          # 7.3.1
ANNEX_SUB_RE = re.compile(r"^([A-Z]\.\d+(?:\.\d+)*)\s+")  # B.2
ANNEX_RE = re.compile(r"^Annex\s+([A-Z])", re.IGNORECASE)
BOILERPLATE_RE = re.compile(r"^.*All rights reserved.*$\n?", re.MULTILINE)


def segment_source_clauses(markdown: str) -> dict[str, str]:
    """Split the source standard by document heading, not by page (Section VI-A).

    Page-wise segmentation lets clauses that share a page contaminate one
    another's embeddings; heading-wise segmentation grounds each clause on its
    own text.
    """
    body = BOILERPLATE_RE.sub("", markdown)
    headings = list(HEADING_RE.finditer(body))
    clauses: dict[str, str] = {}
    for i, m in enumerate(headings):
        heading = m.group(1).strip()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        text = body[m.start():end].strip()
        for pattern in (CLAUSE_RE, ANNEX_SUB_RE, ANNEX_RE):
            hit = pattern.match(heading)
            if hit:
                clauses[hit.group(1)] = text
                break
    return clauses


@dataclass
class TermEntry:
    """One terminology-library record (Section VI-A); the payload lives in configuration."""
    category: str      # terms | modals | phrases | acronyms | values
    source_form: str   # expression in the source standard
    preferred: str     # committee-approved rendering
    alternates: str
    citation: str      # terminology-standard clause the rendering derives from
    note: str          # usage / drift-prevention note


def load_terminology_library(path: Path) -> dict[str, list[TermEntry]]:
    """Load the audited library and index it by the clauses that use each entry.

    Returns {"<standard>|<clause>": [TermEntry, ...]}. The 374 entries and their
    audit history are described in Section VI-A; the entries themselves derive from
    copyrighted terminology standards and are not distributed.
    """
    raise NotImplementedError("Provide the deployment's library loader.")


# ─────────────────────────────────────────────────────────────────────────────
# Constrained generation (paper: Section VI-B)
# ─────────────────────────────────────────────────────────────────────────────

def load_constraints(prompt_path: Path) -> str:
    """Read the generation constraints (repository: prompts/drafting_system.md).

    The constraints are declarative and cover: the five clause categories; the
    freeze on every quantitative requirement value; the JIS Z 8301:2019 modal-force
    map (requirement / recommendation / permission / capability / external
    constraint, each with its prescribed construction and its forbidden
    renderings); the terminology priority chain (terminology standard -> PAUT
    terminology standard -> KG fallback -> committee Open Question) with its
    drift-prevention rules; the paired color-key protocol; and the output schema.
    """
    return prompt_path.read_text(encoding="utf-8")


def build_drafting_request(
    standard: str,
    clause: str,
    source_text: str,
    evidence: dict,
    terms: Sequence[TermEntry],
) -> str:
    """Render the per-clause user message: source text + grounded evidence + library."""

    def fmt(chunks: Sequence[dict], scored: bool = True) -> str:
        if not chunks:
            return "  (none)"
        return "\n".join(
            f"  [{c['citation_id']}] p.{c.get('page')}"
            + (f" Sg={c['Sg']} St={c['St']:.2f}" if scored else "")
            + f"\n  {(c.get('text') or '')[:EXCERPT_CHARS]}"
            for c in chunks
        )

    entities = "\n".join(
        f"  {e['entity']} = {e.get('value') or '(no value)'}  (p.{e.get('page')})"
        for e in evidence["entities"][:20]
    ) or "  (none)"

    library = "\n".join(
        f"  - [{t.category}] \"{t.source_form}\" -> \"{t.preferred}\"  "
        f"({t.citation or 'unsourced'}; {t.note})"
        for t in terms[:20]
    ) or "  (no library entry for this clause)"

    tiers = "\n\n".join(
        f"## {t.id} — {t.role} ({len(evidence['tiers'][t.id])} chunks)\n"
        f"{fmt(evidence['tiers'][t.id])}"
        for t in TIERS
    )

    return (
        f"Draft the national-standard clause corresponding to {standard} Cl.{clause}.\n\n"
        f"## Source clause\n{source_text[:SOURCE_CHARS]}\n\n"
        f"## Specialist entities and values ({len(evidence['entities'])})\n{entities}\n\n"
        f"## Retrieval seed — nearest existing national-standard passages\n"
        f"{fmt(evidence['seed'], scored=False)}\n\n"
        f"{tiers}\n\n"
        f"## Terminology library ({len(terms)} entries; the preferred rendering is binding)\n"
        f"{library}\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Verification and citation traceability (paper: Section VI-B)
# ─────────────────────────────────────────────────────────────────────────────

MARK_RE = re.compile(r"<mark[^>]*background-color\s*:\s*#?([0-9a-fA-F]{6})", re.IGNORECASE)
SOURCE_BLOCK_RE = re.compile(r"^## \[SOURCE\](.*?)(?=^## \[|\Z)", re.MULTILINE | re.DOTALL)
DRAFT_BLOCK_RE = re.compile(r"^## \[DRAFT\](.*?)(?=^## \[|\Z)", re.MULTILINE | re.DOTALL)
# Adjudication-table row: | key n | <mark ...> | source expression | draft expression | ...
ADJUDICATION_ROW_RE = re.compile(
    r"\|\s*key\s*\d+\s*\|\s*<mark[^>]*#?([0-9a-fA-F]{6})[^>]*>[^<]*</mark>\s*\|\s*([^|]+?)\s*\|",
    re.IGNORECASE,
)
CITATION_RE = re.compile(r"([\w　-鿿・\-]+\.json_\d+_\d+_[a-f0-9]{8})")
BACKTICKED_CITATION_RE = re.compile(r"`([\w　-鿿・\-]+\.json_\d+_\d+_[a-f0-9]{8})`")


def count_marks(part: str) -> tuple[int, int]:
    """Color marks on the source side and the draft side of one clause part."""
    src = SOURCE_BLOCK_RE.search(part)
    drf = DRAFT_BLOCK_RE.search(part)
    return (
        len(MARK_RE.findall(src.group(1))) if src else 0,
        len(MARK_RE.findall(drf.group(1))) if drf else 0,
    )


def backfill_source_marks(part: str) -> tuple[str, int]:
    """Recover a broken audit trail from the adjudication table.

    A mark on only one side breaks the trail, so when the source side is unmarked
    the table's (color, source expression) pairs are re-applied to the source text.
    """
    src = SOURCE_BLOCK_RE.search(part)
    if not src or MARK_RE.search(src.group(1)):
        return part, 0

    body, wrapped = src.group(1), 0
    for color, phrase in ADJUDICATION_ROW_RE.findall(part):
        phrase = phrase.strip()
        if not 4 <= len(phrase) <= 200 or "<" in phrase:
            continue
        hit = re.search(re.escape(phrase), body, re.IGNORECASE)
        if not hit:
            continue
        found = hit.group(0)
        body = body[:hit.start()] + f'<mark style="background-color: #{color.lower()};">{found}</mark>' \
            + body[hit.start() + len(found):]
        wrapped += 1

    if not wrapped:
        return part, 0
    return part[:src.start(1)] + body + part[src.end(1):], wrapped


def linkify_citations(part: str, resolver: SourceResolver) -> str:
    """Turn every chunk identifier into a link to the exact source passage.

    Identifier form: {document}.json_{page}_{index}_{hash}; the display form drops
    the file extension. The resolver hides the deployment's viewer, including the
    workaround needed because some office applications strip URI fragments.
    """
    part = BACKTICKED_CITATION_RE.sub(r"\1", part)  # a fenced identifier cannot be linked
    return CITATION_RE.sub(
        lambda m: f"[{m.group(1).replace('.json', '', 1)}]({resolver.uri(m.group(1))})",
        part,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-clause driver
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DraftResult:
    clause: str
    part: str
    source_marks: int
    draft_marks: int
    regenerated: bool = False
    backfilled: int = 0
    notes: list[str] = field(default_factory=list)


REGENERATION_NOTICE = (
    "\n\n## Regeneration\n"
    "The previous output left the source side of the color key unmarked, which breaks the "
    "audit trail. Mark both sides this time: every color must pair one source passage with "
    "its drafted counterpart."
)


def draft_clause(
    standard: str,
    clause: str,
    source_text: str,
    seed_chunk_ids: Sequence[str],
    terms: Sequence[TermEntry],
    corpus: Corpus,
    generator: Generator,
    resolver: SourceResolver,
    constraints: str,
) -> DraftResult:
    """Retrieve, generate under constraints, verify (paper: Section VI-B)."""
    evidence = retrieve_evidence(standard, clause, seed_chunk_ids, corpus,
                                title=_title_of(source_text))
    request = build_drafting_request(standard, clause, source_text, evidence, terms)

    part = _strip_code_fence(generator.complete(constraints, request))
    result = DraftResult(clause, part, *count_marks(part))

    # One regeneration attempt for an asymmetric color key, then mechanical backfill.
    if result.draft_marks and not result.source_marks:
        retry = _strip_code_fence(generator.complete(constraints, request + REGENERATION_NOTICE))
        src_marks, drf_marks = count_marks(retry)
        result.regenerated = True
        if src_marks:
            result.part, result.source_marks, result.draft_marks = retry, src_marks, drf_marks
        else:
            result.part, result.backfilled = backfill_source_marks(retry or part)
            result.source_marks, result.draft_marks = count_marks(result.part)
            if not result.backfilled:
                result.notes.append("color key asymmetric — needs manual review")

    result.part = linkify_citations(result.part, resolver)
    return result


def _title_of(source_text: str) -> str:
    for line in source_text.splitlines():
        if line.startswith("#"):
            return re.sub(r"^#+\s*[\d.]*\s*", "", line).strip()
    return ""


def _strip_code_fence(text: str) -> str:
    """Drop a wrapping ``` fence, which some models add around Markdown output."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
#
# The concrete Corpus / Generator / SourceResolver implementations and the
# document-assembly stage (Section VI-C) are deployment-specific; credentials come from
# NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD and the model API key.
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Draft national-standard clauses from a source standard.")
    parser.add_argument("--standard", required=True, help="source-standard identifier, e.g. ISO13588")
    parser.add_argument("--clause", nargs="*", default=[], help="clause numbers (default: all)")
    parser.add_argument("--new-only", action="store_true", help="skip clauses already drafted")
    parser.add_argument("--out", type=Path, default=Path("parts"), help="output directory")
    args = parser.parse_args()

    corpus, generator, resolver = build_services(os.environ)
    constraints = load_constraints(Path("prompts/drafting_system.md"))
    library = load_terminology_library(Path("config/terminology_library.yaml"))
    sources = segment_source_clauses(load_source_markdown(args.standard))

    clauses = args.clause or list(sources)
    args.out.mkdir(parents=True, exist_ok=True)
    for clause in clauses:
        out_path = args.out / f"{args.standard}-{clause.replace('.', '-').lower()}.md"
        if args.new_only and out_path.exists():
            continue
        result = draft_clause(
            standard=args.standard,
            clause=clause,
            source_text=sources.get(clause, ""),
            seed_chunk_ids=load_retrieval_seed(args.standard, clause),
            terms=library.get(f"{args.standard}|{clause}", []),
            corpus=corpus,
            generator=generator,
            resolver=resolver,
            constraints=constraints,
        )
        out_path.write_text(result.part, encoding="utf-8")
        print(f"{args.standard} Cl.{clause}: marks {result.source_marks}/{result.draft_marks}"
              + (" [regenerated]" if result.regenerated else "")
              + (f" [backfilled {result.backfilled}]" if result.backfilled else "")
              + ("  " + "; ".join(result.notes) if result.notes else ""))


def build_services(env) -> tuple[Corpus, Generator, SourceResolver]:
    """Wire the knowledge graph, the model client, and the citation resolver."""
    raise NotImplementedError("Provide the deployment's service factory.")


def load_source_markdown(standard: str) -> str:
    """Return the Markdown text of the source standard, supplied by the deployment."""
    raise NotImplementedError("Provide the deployment's source loader.")


def load_retrieval_seed(standard: str, clause: str) -> list[str]:
    """Cached top existing-national-standard chunk identifiers for this clause (Section VI-A)."""
    raise NotImplementedError("Provide the deployment's seed loader.")


if __name__ == "__main__":
    main()
