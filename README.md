<!-- Generated for upload to https://github.com/Kamapond/Ueno-lab2 (commit to repo ROOT).
     Replace all [PLACEHOLDER] items before committing. After minting the Zenodo DOI (see
     SECURITY_AUDIT_CHECKLIST.md / repo steps), paste the DOI badge line at the top. -->

# Knowledge-Graph-Driven RAG for Accelerating International Standards Alignment

<!-- [PLACEHOLDER: after Zenodo release, add the DOI badge, e.g.]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference implementation for the IEEE Access article *“Knowledge-Graph-Driven RAG for Accelerating
International Standards Alignment.”* The study builds a three-stage pipeline — a Plain large-language-model
(LLM) baseline, retrieval-augmented generation (RAG), and graph-based RAG (GraphRAG) over a domain
knowledge graph — and evaluates specialist ability on a 904-item certification benchmark with a
two-parameter item response theory (IRT) model. This repository contains the code for the IRT estimation,
the RAG pipeline, and the structure-aware knowledge-graph (GraphRAG) construction and retrieval.

## Repository structure ↔ paper mapping

<!-- REWRITTEN 2026-07-30, updated 2026-08-02 to match the article appendixes (A–D; the drafting
     appendix was deleted, so jis_drafting/ is described in §VI). preprocessing/ is NOT released. The old folder names
     mirrored the Japanese source manuscript's appendix labels ("A.1 IRT_2PL Binomial", "F.2 Neo4j
     Ingenstion", …); they are replaced by semantic, space-free names so that (a) tag-pinned permalinks
     need no %20 escaping and (b) a change to the article's appendix lettering never churns the repo.
     ACTION REQUIRED on the live repo: rename the existing folders to the layout below, and DELETE the
     existing `OCR` folder (author decision 2026-08-02 — preprocessing is neither released nor described
     in the article; a folder the article never mentions has no reader value).
     DATA POLICY 2026-08-02 (author): the repository is CODE ONLY. The earlier note promising the
     per-category ICC / θ-posterior data here is removed — those data are request-only, matching
     APPENDIX D of the article. This also removes the conflict with .gitignore, which blocks
     *.csv / *.json / *.xlsx / data/ / outputs/. -->

Each module corresponds to the part of the article that describes it. Cite files by tag-pinned permalink, e.g.
`https://github.com/Kamapond/Ueno-lab2/tree/v1.0.0/rag`.

```
.
├── irt/                            → Appendix A
│   ├── model.py                    2PL-binomial model definition
│   ├── sampling.py                 NUTS/MCMC posterior sampling (TABLE V settings)
│   └── aggregate_icc.py            aggregate ICC + latent-ability plot
├── baseline/                       → Section III-E, Condition 1 (Plain LLM)
│   ├── prompts.py                  baseline system prompt
│   ├── extraction.py               regex extraction of the selected option
│   └── run_batch.py                batch inference harness (CLI)
├── rag/                            → Appendix B
│   ├── vector_store.py             chunking (500/150) + deterministic-ID ingestion
│   ├── prompts.py                  grounding prompt (TABLE VI)
│   └── chain.py                    LCEL retrieve-and-generate chain
├── graphrag/                       → Appendix C
│   ├── schema.py                   25 entity / 103 relation types (TABLE VII, VIII)
│   ├── extraction.py               heading-aware parsing + LLM graph extraction
│   ├── ingestion.py                Neo4j ingestion (tree · labels · relations · GT-Link)
│   ├── retrieval.py                two-axis (Sg, St) scoring + Skyline selection
│   └── generation.py               scored-context rendering + answer generation
└── jis_drafting/                   → Section VI
    └── draft_jis_clauses.py        segment · retrieve · generate · verify
```

| Repository folder | Described in | Purpose |
|---|---|---|
| `irt/` | Appendix A | 2PL-binomial IRT model definition (NumPyro), NUTS/MCMC sampling, aggregate-ICC computation and plotting |
| `rag/` | Appendix B | Chunking + Chroma vector-store construction, LCEL retrieve-and-generate chain, prompt definitions |
| `baseline/` | §III-E, Condition 1 | Plain-LLM batch API inference; regex-based extraction of the selected option |
| `graphrag/` | Appendix C | LLM graph extraction under the domain schema, Neo4j ingestion (Document→Section→Chunk tree + GT-Links), two-axis scoring (*Sg*, *St*), Skyline (Pareto-optimal) selection, GraphRAG prompt |
| `jis_drafting/` | Section VI | ISO clause segmentation, terminology library + audit, constrained generation, verification (`draft_jis_clauses.py`) |

> Note: the knowledge-graph schema (25 entity types / 103 relation types) and the prompt-design table are
> reproduced inline in the article (Appendix C and Appendix B respectively) and need no code folder.

> **These are reference implementations, not turnkey scripts.** A small number of functions mark
> deployment boundaries and deliberately raise `NotImplementedError`, because supplying them would
> require site-specific credentials, local paths, or copyrighted source text:
> `rag/vector_store.py::get_all_ids`, `graphrag/retrieval.py::extract_search_entities`, and
> `jis_drafting/draft_jis_clauses.py::{build_services, load_source_markdown, load_retrieval_seed,
> load_terminology_library}`. Supply your own implementations to run them. Every algorithm the article
> describes — IRT estimation, chunking, hybrid scoring, Skyline selection, constrained generation,
> verification — is published in full.

## Requirements

- Python 3.13
- Install dependencies:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration (API keys & database)

Credentials are read from environment variables — **never commit them**. Copy `.env.example` to `.env`
and fill in your own values:

```
OPENAI_API_KEY=...
NEO4J_URI=...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
```

`.env` is git-ignored (see `.gitignore`). Do not paste keys into source files.

## Data availability

**This repository contains code only. No experimental data are distributed here.**

The benchmark questions (JSNDI certification examinations) and the knowledge-source documents (JIS/NDIS
standards and JSNDI technical texts) are **copyrighted and are not distributed**.

Model-response data — including the per-category 2PL item-characteristic-curve and latent-ability (θ)
posterior data underlying the aggregate results reported in the article — are **available from the
corresponding author on reasonable request**.

The `.gitignore` in this repository blocks `*.csv`, `*.json`, `*.xlsx`, `data/` and `outputs/` by design,
so that no experimental or copyrighted material can be committed inadvertently.

## Citation

If you use this code, please cite the article and the archived snapshot:

```bibtex
@article{Matsuzono2026GraphRAG,
  author  = {Matsuzono, Shinichi and Ueno, Tsuyoshi},
  title   = {Knowledge-Graph-Driven RAG for Accelerating International Standards Alignment},
  journal = {IEEE Access},
  year    = {2026},
  note    = {[PLACEHOLDER: volume, pages, DOI on acceptance]}
}

@software{Matsuzono2026Code,
  author  = {Matsuzono, Shinichi and Ueno, Tsuyoshi},
  title   = {Knowledge-Graph-Driven RAG for Accelerating International Standards Alignment (code)},
  year    = {2026},
  version = {v1.0.0},
  doi     = {[PLACEHOLDER: 10.5281/zenodo.XXXXXXX]},
  url     = {https://github.com/Kamapond/Ueno-lab2}
}
```

## License

Released under the [MIT License](LICENSE). The copyrighted benchmark and knowledge-source data are **not**
covered by this license and are not distributed here.

## Contact

Shinichi Matsuzono — ORCID [0009-0007-6262-6032](https://orcid.org/0009-0007-6262-6032),
e-mail [PLACEHOLDER: address].
