"""
prompts.py — system prompt for the Plain-LLM baseline
(paper: Section III-E, Condition 1; repository module `baseline/`).

The prompt fixes two things: the role the model is to adopt, and a strict output format. The
format matters more than the role — separating the chosen option from the justification, and
placing the option first, is what makes the responses machine-extractable downstream
(`extraction.py`) and therefore makes automated scoring of 904 items feasible.

The prompt is reproduced verbatim in Japanese, the language of the benchmark. No external
knowledge is supplied under this condition: answers rest solely on parametric knowledge
acquired during pre-training.
"""

from __future__ import annotations

SYSTEM_PROMPT = """
あなたは専門的な技術認証試験の問題に解答します。
指示された問題に対し、以下のルールに従って解答してください。
出力は次の形式に厳密に従ってください。

1. 1つの問題項番は、[1]...[n]の複数の問題と解答で構成されます。
`[解答群]`に続く[1]...[n]に対応するそれぞれの4者択一の選択肢から、
必ず、最も正しいと思われる解答を、1つだけ選択してください。
2. まず、解答の選択肢だけを提示してください。形式は `解答: ` の後に続けて記述します。
3. その後、改行を挟んでから、その解答を選んだ理由を具体的に説明してください。
"""
