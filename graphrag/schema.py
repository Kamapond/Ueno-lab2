"""
schema.py — domain extraction schema for the knowledge graph
(paper: Appendix C, TABLE VII and TABLE VIII; repository module `graphrag/`).

Constraining extraction to a fixed, domain-specific vocabulary is what separates a usable
specialist graph from a noisy one. Left unconstrained, a general extractor invents near-
duplicate node and relation labels for the same concept, and the graph stops being traversable.
The schema below fixes 25 entity types in 6 categories and 103 relation types in 8 categories,
tailored to national standards and non-destructive evaluation.

The relation vocabulary is deliberately fine-grained on the axes that carry normative meaning —
requirement and logic (17 types) and measurement and value (18 types) — because in standards
text the difference between "shall be" and "is recommended", or between a maximum and a
tolerance, is the content rather than a nuance of phrasing.

"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Entity types — 25 in 6 categories (paper: TABLE VII)
# ─────────────────────────────────────────────────────────────────────────────
ENTITY_TYPES: dict[str, list[str]] = {
    "Document structure": ["Standard", "Document", "Section", "Table", "Figure"],
    "Inspection target": ["Weld", "Material", "TestObject", "Component", "Defect"],
    "Inspection equipment": ["Device", "Probe", "Wedge", "Block", "Accessory"],
    "Method / Process": ["Method", "Process", "Test"],
    "Data": ["Parameter", "Value", "Signal"],
    "Concept": ["Requirement", "Condition", "Concept", "Actor"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Relation types — 103 in 8 categories (paper: TABLE VIII)
# ─────────────────────────────────────────────────────────────────────────────
RELATION_TYPES: dict[str, list[str]] = {
    "Structure": [
        "INCLUDES", "HAS_PART", "PART_OF", "HAS_SECTION", "HAS_CLAUSE", "HAS_NOTE",
        "HAS_FIGURE", "HAS_TABLE", "SUBTYPE_OF", "INSTANCE_OF", "COMPOSED_OF",
        "DERIVED_FROM",
    ],
    "Definition / Reference": [
        "DEFINES", "DEFINED_AS", "DESCRIBES", "DESCRIBED_IN", "CITES", "REFERENCES",
        "REFERENCED_IN", "ILLUSTRATES", "ILLUSTRATED_BY", "SAME_AS", "ALIAS_OF",
        "ABBREVIATION_OF",
    ],
    "Use / Equipment": [
        "USES", "USED_FOR", "USED_IN", "USED_WITH", "REQUIRES_TOOL", "EQUIPPED_WITH",
        "MOUNTED_ON", "CONNECTED_TO", "COMBINED_WITH", "COMPATIBLE_WITH", "REPLACES",
        "SUBSTITUTED_BY",
    ],
    "Scope / Applicability": [
        "APPLIES_TO", "APPLICABLE_FOR", "NOT_APPLICABLE_TO", "COVERS", "EXCLUDES",
        "TARGETS", "LIMITED_TO", "VALID_FOR", "SUITABLE_FOR", "INTENDED_FOR",
    ],
    "Requirement / Logic": [
        "REQUIRES", "MUST_BE", "MUST_HAVE", "MUST_NOT", "SHOULD_BE", "RECOMMENDED",
        "OPTIONAL", "PERMITTED", "PROHIBITED", "CONDITION_FOR", "DEPENDS_ON",
        "CONFORMS_TO", "SATISFIES", "VIOLATES", "IF_THEN", "CAUSES", "RESULTS_IN",
    ],
    "Measurement / Value": [
        "MEASURES", "CALCULATED_BY", "ESTIMATED_FROM", "HAS_VALUE", "HAS_UNIT",
        "HAS_PARAMETER", "HAS_PROPERTY", "HAS_TOLERANCE", "HAS_RANGE", "HAS_MINIMUM",
        "HAS_MAXIMUM", "GREATER_THAN", "LESS_THAN", "EQUAL_TO", "WITHIN_RANGE",
        "EXCEEDS", "CONVERTED_TO", "RATIO_OF",
    ],
    "Process": [
        "PERFORMED_BY", "CONDUCTED_AT", "MANUFACTURED_BY", "CALIBRATED_BY", "ADJUSTED_TO",
        "CHECKED_BY", "VERIFIED_BY", "APPROVED_BY", "RECORDED_IN", "REPORTED_TO",
        "FOLLOWS", "PRECEDES",
    ],
    "NDI physics": [
        "DETECTS", "INDICATES", "GENERATES_SIGNAL", "PROPAGATES_IN", "REFLECTS_AT",
        "ATTENUATES_IN", "LOCATED_AT", "ORIENTED_IN", "DISPLAYED_ON", "VISIBLE_TO",
    ],
}

ALLOWED_NODES: list[str] = [t for types in ENTITY_TYPES.values() for t in types]
ALLOWED_RELATIONSHIPS: list[str] = [t for types in RELATION_TYPES.values() for t in types]

# Structural relations that are created during ingestion rather than by the extractor:
# the document tree, chunk ordering, and the graph-tree link.
STRUCTURAL_RELATIONSHIPS = ["HAS_CHILD", "NEXT", "MENTIONS"]

assert len(ALLOWED_NODES) == 25, f"expected 25 entity types, got {len(ALLOWED_NODES)}"
assert len(ALLOWED_RELATIONSHIPS) == 103, (
    f"expected 103 relation types, got {len(ALLOWED_RELATIONSHIPS)}"
)
