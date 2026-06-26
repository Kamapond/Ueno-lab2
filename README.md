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

## Repository structure ↔ paper appendix mapping

| Repository folder | Paper appendix | Purpose |
|---|---|---|
| `A.1 IRT_2PL Binomial` | Appendix A.1 | 2PL-binomial IRT model definition (NumPyro) |
| `A.2 MCMC` | Appendix A.2 | NUTS / MCMC posterior sampling |
| `A.3 ICC calculation` | Appendix A.3 | Aggregate item characteristic curve (ICC) computation |
| `A.4 ICC Drawing` | Appendix A.4 | ICC / latent-ability (θ) plotting |
| `D.2 Answer extraction logic` | Appendix D.2 | Regex-based answer extraction from LLM output |
| `D.3 API Batch Processing` | Appendix D.3 | Batch API inference (Plain baseline) |
| `E.1 Chroma Vector` | Appendix E.1 | Chunking + Chroma vector store construction |
| `E.2 RAG_LCEL` | Appendix E.2 | LCEL retrieve-and-generate chain (RAG) |
| `F.1 Extraction` | Appendix F.1 | LLM graph extraction (entities / relations) |
| `F.2 Neo4j Ingenstion` | Appendix F.2 | Neo4j ingestion (Document→Section→Chunk tree + GT-Links) |
| `F.3 Hybrid Scoring` | Appendix F.3 | Two-axis scoring (graph score Sg, text score St) |
| `F.4 Skyline Ranking` | Appendix F.4 | Skyline (Pareto-optimal) context selection |
| `F.5 GraphRAG Prompt` | Appendix F.5 | GraphRAG answer-generation prompt |
| `OCR` | (preprocessing) | Mistral OCR → Markdown conversion of source standards (not a numbered appendix) |

> Notes: Appendix A.1–A.4 belong to *Appendix A* in the paper. Appendix B (KG schema) is a table, and the
> system prompt (Appendix D.1) is reproduced inline in the article; neither needs a code folder.

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

The benchmark questions (JSNDI certification examinations) and the knowledge-source documents (JIS/NDIS
standards and JSNDI technical texts) are **copyrighted and are not included** in this repository.
Model-response data are available from the corresponding author on reasonable request.

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

[PLACEHOLDER: corresponding author name], [PLACEHOLDER: e-mail], ORCID [PLACEHOLDER: 0000-0000-0000-0000].
