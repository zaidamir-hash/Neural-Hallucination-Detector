"""Seven scalar features for the primary representation.

Computes the lexical and numeric coverage features from raw text.
Inputs use the same whitespace normalization as the encoder, followed by
separate lightweight token normalization for lexical matching.
"""
from __future__ import annotations

import re
from typing import List, Set
from hallucination_detector.features.preprocessing import normalize_text


def _content_tokens(text: str) -> List[str]:
    """Lowercased punctuation-stripped tokens for lexical matching."""
    toks = normalize_text(text).split()
    cleaned = [re.sub(r"(^\W+|\W+$)", "", t).lower() for t in toks]
    return [t for t in cleaned if t]


def _content_token_set(text: str) -> Set[str]:
    return set(_content_tokens(text))


def _numeric_tokens(text: str) -> List[str]:
    """Tokens containing at least one digit."""
    toks = normalize_text(text).split()
    return [t for t in toks if re.search(r"\d", t)]


def _numeric_token_set(text: str) -> Set[str]:
    return set(_numeric_tokens(text))


def compute_answer_content_coverage(answer: str, context: str) -> float:
    """Fraction of answer content tokens found in context."""
    a_tokens = _content_token_set(answer)
    if not a_tokens:
        return 0.0
    c_tokens = _content_token_set(context)
    covered = sum(1 for t in a_tokens if t in c_tokens)
    return covered / len(a_tokens)


def compute_answer_numeric_coverage(answer: str, context: str) -> float:
    """Fraction of answer numeric tokens found in context."""
    a_num = _numeric_token_set(answer)
    if not a_num:
        return 0.0
    c_tokens = _content_token_set(context)
    covered = sum(1 for t in a_num if t in c_tokens)
    return covered / len(a_num)


def compute_answer_has_numeric(answer: str) -> float:
    """1.0 if the answer contains a numeric token, else 0.0."""
    return 1.0 if _numeric_tokens(answer) else 0.0
