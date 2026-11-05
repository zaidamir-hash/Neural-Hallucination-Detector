"""Frozen sentence-transformer encoder wrapper.

Provides a reusable, deterministic, CPU-compatible wrapper around
`sentence-transformers/all-MiniLM-L6-v2` that produces 384-dimensional
L2-normalized embeddings with all parameters frozen.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class FrozenEncoder:
    """A frozen sentence-transformer encoder.

    All model parameters are set to ``requires_grad=False`` and the model
    is placed in ``eval`` mode.  No gradient tracking occurs during
    encoding.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._batch_size = batch_size
        self._model_name = model_name

        self._model = SentenceTransformer(model_name, device=device)

        # Freeze all parameters
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.eval()

        self._revision: Optional[str] = None
        try:
            self._revision = getattr(self._model, "_model_card_vars", {}).get(
                "model_revision", None
            )
        except Exception:
            pass

        self._st_version: Optional[str] = None
        try:
            import sentence_transformers as st
            self._st_version = st.__version__
        except Exception:
            pass

        logger.info(
            "Loaded frozen encoder %s (device=%s, revision=%s, st_version=%s)",
            model_name, device, self._revision, self._st_version,
        )

    # ------------------------------------------------------------------
    # Public API
    # # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    @property
    def device(self) -> str:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def revision(self) -> Optional[str]:
        return self._revision

    @property
    def sentence_transformers_version(self) -> Optional[str]:
        return self._st_version

    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts into L2-normalized embeddings.

        Args:
            texts: List of non-empty strings.
            batch_size: Override the default batch size.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            ndarray of shape ``(len(texts), 384)`` with float32 values.

        Raises:
            ValueError: If ``texts`` is empty or contains non-string items.
            RuntimeError: If the output shape is unexpected or contains
                non-finite values.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list of strings")
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"texts[{i}] is {type(t).__name__}, expected str")

        bs = batch_size or self._batch_size

        with torch.no_grad():
            embeddings: np.ndarray = self._model.encode(
                texts,
                batch_size=bs,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        # Validate shape
        expected_shape = (len(texts), EMBEDDING_DIM)
        if embeddings.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected embedding shape: {embeddings.shape}, expected {expected_shape}"
            )

        # Validate finiteness
        if not np.all(np.isfinite(embeddings)):
            raise RuntimeError("Embedding output contains non-finite values")

        return embeddings.astype(np.float32)

    def get_tokenizer(self):
        """Return the underlying tokenizer for chunking."""
        return self._model.tokenizer
