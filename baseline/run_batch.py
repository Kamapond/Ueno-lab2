"""
run_batch.py — batch inference for the Plain-LLM baseline
(paper: Section III-E, Condition 1; repository module `baseline/`).

Reads the benchmark index, sends each item to the model under test, and stores the extracted
option alongside the full response. This is the measurement harness for Step 1: no external
knowledge is injected, so the resulting accuracy reflects parametric knowledge alone and
becomes the baseline against which the RAG and GraphRAG conditions are compared.

Token counts are recorded per item because the cost-efficiency analysis of Section V-E is
computed from them.

Deliberate abstractions in this published version
  * The API key is read from the environment; no credential is embedded.
  * Benchmark locations are supplied on the command line. The benchmark items themselves are
    copyrighted certification-examination material and are NOT distributed with this
    repository (see the Data availability section of the README).
  * The experiment sweeps eight models under five option-order patterns; this script runs one
    model over one index file, and the sweep is driven externally.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import openai
import pandas as pd
from tqdm import tqdm

from .extraction import extract_choice
from .prompts import SYSTEM_PROMPT


def run(
    index_path: Path,
    sheet_name: str,
    json_dir: Path,
    output_path: Path,
    model_name: str,
) -> Path:
    """Answer every item listed in the index and write the results table.

    Args:
        index_path:  spreadsheet listing the benchmark items (one row per item, column "file").
        sheet_name:  worksheet holding that list.
        json_dir:    directory of per-item JSON documents, each with a "markdown" field.
        output_path: destination spreadsheet.
        model_name:  model identifier passed straight to the API.
    """
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    df = pd.read_excel(index_path, sheet_name=sheet_name)

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        filename = row.get("file")
        with open(json_dir / f"{filename}.json", "r", encoding="utf-8") as f:
            question_markdown = json.load(f).get("markdown")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_markdown},
        ]

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )

            answer_full = response.choices[0].message.content.strip()

            df.loc[index, "Choice"] = extract_choice(answer_full)
            df.loc[index, "Answer"] = answer_full
            df.loc[index, "Model"] = model_name

            # Retained for the cost-efficiency analysis (Section V-E).
            usage = response.usage
            df.loc[index, "InputTokens"] = usage.prompt_tokens
            df.loc[index, "OutputTokens"] = usage.completion_tokens

        except Exception as exc:  # keep the sweep running; failures stay visible in the output
            df.loc[index, "Answer"] = f"Error: {exc}"

    df.to_excel(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--index", type=Path, required=True, help="benchmark index spreadsheet")
    parser.add_argument("--sheet", default="Sheet1", help="worksheet name")
    parser.add_argument("--json-dir", type=Path, required=True, help="per-item JSON directory")
    parser.add_argument("--output", type=Path, required=True, help="destination spreadsheet")
    parser.add_argument("--model", required=True, help="model identifier, e.g. gpt-5.1")
    args = parser.parse_args()

    out = run(args.index, args.sheet, args.json_dir, args.output, args.model)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
