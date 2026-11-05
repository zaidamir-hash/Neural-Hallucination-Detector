"""Tokenizer-aware context chunking.

Splits a context string into chunks that each fit within the encoder's
token limit, with a configurable overlap.  Uses the encoder's own
tokenizer so that chunk boundaries respect real token limits.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol
from hallucination_detector.features.preprocessing import normalize_text


class SupportsTokenize(Protocol):
    """Minimal protocol for a tokenizer that can encode text."""

    def __call__(
        self, text: str, return_tensors: Optional[str] = None, **kwargs: Any
    ) -> Any: ...


def chunk_context(
    context: str,
    tokenizer: SupportsTokenize,
    max_chunk_tokens: int = 192,
    overlap_tokens: int = 32,
) -> List[str]:
    """Split *context* into overlapping token-aware chunks.

    Each chunk is guaranteed to contain at most ``max_chunk_tokens``
    tokens (as counted by *tokenizer*).  Successive chunks overlap by
    ``overlap_tokens`` tokens.

    Rules:
      - Empty or whitespace-only context raises ``ValueError``.
      - Short contexts that fit in a single chunk are returned as-is.
      - The final partial chunk is always retained.
      - Chunk order is deterministic.
      - No infinite loop: the stride is always positive.

    Special tokens: The tokenizer's ``[CLS]`` and ``[SEP]`` tokens are
    **not** added by this function — we tokenize without special tokens
    to count content tokens only.  This means each chunk's effective
    token count when later passed through the full model pipeline will
    include special tokens but remain within the model's window because
    ``max_chunk_tokens`` is set below the model's absolute limit.

    Args:
        context: The text to chunk.
        tokenizer: A HuggingFace-style tokenizer callable.
        max_chunk_tokens: Maximum number of content tokens per chunk.
        overlap_tokens: Number of overlapping tokens between consecutive
            chunks.

    Returns:
        A list of chunk strings (at least one).

    Raises:
        ValueError: If context is empty/whitespace, or overlap >= max.
    """
    context = normalize_text(context)
    if not context:
        raise ValueError("context must be a non-empty string")

    if overlap_tokens >= max_chunk_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be < max_chunk_tokens ({max_chunk_tokens})"
        )

    # Tokenize without special tokens to get true content token count
    encoded = tokenizer(context, add_special_tokens=False)
    input_ids: List[int] = encoded["input_ids"]
    total_tokens = len(input_ids)

    # Short context — fits in one chunk
    if total_tokens <= max_chunk_tokens:
        return [context]

    chunks: List[str] = []
    stride = max_chunk_tokens - overlap_tokens
    start = 0

    while start < total_tokens:
        end = min(start + max_chunk_tokens, total_tokens)
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end >= total_tokens:
            break
        start += stride

    if not chunks:
        # Fallback: should not happen, but guard against edge cases
        return [context]

    return chunks


def chunk_contexts_batch(
    contexts: List[str],
    tokenizer: SupportsTokenize,
    max_chunk_tokens: int = 192,
    overlap_tokens: int = 32,
) -> List[List[str]]:
    """Chunk multiple contexts. Returns a list of chunk-lists."""
    return [
        chunk_context(c, tokenizer, max_chunk_tokens, overlap_tokens)
        for c in contexts
    ]
