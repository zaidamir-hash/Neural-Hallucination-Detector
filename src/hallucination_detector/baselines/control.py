"""Answer-only style/length feature extraction; no question/context inputs."""
from typing import List
import unicodedata

import numpy as np

from hallucination_detector.data.audit import sentence_count, tokens_whitespace, numeric_tokens

CONTROL_SCHEMA = {
    "version": "1.0.0",
    "features": ["character_count", "whitespace_token_count", "sentence_count",
                 "numeric_token_presence", "numeric_token_count", "punctuation_count"],
    "input": "original candidate_answer strings from processed rows; no whitespace normalization",
    "sentence_rule": "nonempty segments split on . ? !",
    "numeric_rule": "whitespace tokens containing at least one digit",
    "punctuation_rule": "count characters whose Unicode category begins with P",
}


def answer_surface_features(answers: List[str]) -> np.ndarray:
    """Extract six surface features using only candidate answer strings."""
    rows = []
    for answer in answers:
        if not isinstance(answer, str):
            raise TypeError("Each candidate answer must be a string")
        numbers = len(numeric_tokens(answer))
        rows.append([len(answer), len(tokens_whitespace(answer)), sentence_count(answer),
                     int(numbers > 0), numbers,
                     sum(unicodedata.category(char).startswith("P") for char in answer)])
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(CONTROL_SCHEMA["features"]))
