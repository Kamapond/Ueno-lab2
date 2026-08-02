"""
prompts.py — grounding prompt for the retrieval-augmented conditions
(paper: Appendix B, TABLE VI; repository module `rag/`).

The prompt does three jobs: it fixes the assistant's role, it binds every answer to the
retrieved context, and it fixes an output format that can be scored automatically.

The grounding marker is the load-bearing part. When the retrieved context contains no basis for
an answer, the model must emit `Context_Unspecified` on the first line before falling back to
its own knowledge. That single token makes it possible to separate, after the fact, answers
that were genuinely grounded in retrieved evidence from answers that were produced from
parametric knowledge — which is what allows the effect of knowledge injection to be measured
rather than assumed, and hallucination under retrieval to be quantified.

The same prompt is used for the GraphRAG condition, so that the two retrieval conditions differ
only in how context is selected and never in how it is presented or scored.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

GROUNDING_MARKER = "Context_Unspecified"

SYSTEM_PROMPT = (
    "あなたは専門的な技術認証試験の問題に解答するGraphRAGアシスタントである。\n"
    "それぞれの問題の解答を行う際は必ず、出題内容に関連する情報を提供する\n"
    "『コンテキスト』の内容を参照して、日本語で記述すること。\n"
    "解答に際して、提供されたコンテキストの内容にまったく関連が無いと判断した場合は、\n"
    "最初の行で必ず 'Context_Unspecified' と明記したうえで、\n"
    "その後にあなたの一般知識による最善の解答を提示する。\n"
    "出力は厳密に次の形式に従うこと：\n\n"
    "  1) 先頭行に '解答: a' のように a/b/c/d のいずれか1文字のみを半角で記す。\n"
    "  2) 2行目以降に、その選択理由を箇条書きで簡潔に示す（可能なら根拠のページを [p.数字] で併記）。\n"
    "禁止事項：前置き・要約・挨拶・ラベル変更・フォーマットの追加は行わない。"
)

RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",
     "質問: {question}\n\n"
     "コンテキスト:\n{context}\n\n"
     "ルール:\n"
     "- すべての回答は必ず『提供されたコンテキスト』に基づくこと。\n"
     "- 4者択一から **1つだけ** 選ぶこと（a/b/c/d の半角1文字）。\n"
     "- コンテキストに根拠が無い場合は、その設問の回答直前に 'Context_Unspecified' を出力する。\n\n"
     "出力フォーマット（厳守）:\n"
     "【単一設問の場合】\n"
     "解答: a\n"
     "- 理由1… [p.12]\n\n"
     "【複数小問（[1]〜[n]）の場合】\n"
     "[解答群]\n"
     "解答[1]: a\n"
     "- 理由...\n"
     "解答[2]: c\n"
     "- 理由...\n"
     ),
])
