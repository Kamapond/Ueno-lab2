"""
extraction.py — regular-expression extraction of the selected option
(paper: Section III-E, Condition 1; repository module `baseline/`).

Even under an explicit format instruction, models vary in how they mark their choice: "[1](a)",
"解答: A", or a bare leading "(a)". This module absorbs that variation and reduces each response
to the lower-case letters that can be compared against the answer key.

Two passes, in order:
  1. the multi-item pattern, which also covers a single item written in the "解答:" form;
  2. a fallback for a response that opens directly with the option letter.

A response that matches neither returns the sentinel "N/A" rather than a guess, so that
unparseable responses stay visible in the results instead of being silently scored.
"""

from __future__ import annotations

import re

# "[1](a)" / "[1] a" style, or "解答: a" (optionally a comma-separated run for sub-items).
MULTI_PATTERN = r"\[\d+\]\s*\(?([a-zA-Z])\)?|解答:\s*([a-zA-Z,\s]+)"

# A response opening directly with the option letter, e.g. "(a) ..." or "a ...".
SINGLE_PATTERN = r"^\s*\(?([a-zA-Z])\)?"

NOT_AVAILABLE = "N/A"


def extract_choice(text: str) -> str:
    """Return the selected option(s) as lower-case letters, or "N/A" if none can be read.

    For an item with sub-questions the letters are concatenated in order, so "[1](a) [2](c)"
    yields "ac" and can be compared directly against the concatenated key.
    """
    matches = re.findall(MULTI_PATTERN, text)

    if matches:
        # findall returns one tuple per match, with the unmatched alternative empty.
        choices_str = "".join(item for tpl in matches for item in tpl if item)
        choices = re.sub(r"[,\s]", "", choices_str)
        if choices:
            return choices.lower()

    single_match = re.match(SINGLE_PATTERN, text)
    if single_match and single_match.groups():
        return single_match.group(1).lower()

    return NOT_AVAILABLE
