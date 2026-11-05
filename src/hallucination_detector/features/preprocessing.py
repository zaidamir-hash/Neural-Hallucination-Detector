"""Deterministic normalization for Phase 3 model inputs only."""

PREPROCESSING_VERSION = "2.0.0"


def normalize_text(text: str) -> str:
    """Strip outer whitespace and collapse Unicode whitespace to ASCII spaces."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    return " ".join(text.split())
