"""Embedding generation with deterministic caching.

Generates and caches question, answer, and context-chunk embeddings
using the frozen encoder.  Caches are stored as NumPy arrays with a
JSON metadata sidecar.  Incompatible caches are rejected.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hallucination_detector.features.encoder import FrozenEncoder

logger = logging.getLogger(__name__)

PREPROCESSING_VERSION = "1.0.0"
CACHE_FORMAT_VERSION = "1.0.0"


def _compute_checksum(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _text_hash(text: str) -> str:
    """Deterministic hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_cache_metadata(
    dataset_checksum: str,
    split_manifest_checksum: str,
    encoder: FrozenEncoder,
    chunk_size: int,
    chunk_overlap: int,
    split: str,
    row_count: int,
    feature_dim: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the metadata dict for a feature/embedding cache."""
    meta: Dict[str, Any] = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset_checksum": dataset_checksum,
        "split_manifest_checksum": split_manifest_checksum,
        "encoder_name": encoder.model_name,
        "encoder_revision": encoder.revision,
        "sentence_transformers_version": encoder.sentence_transformers_version,
        "embedding_dim": encoder.embedding_dim,
        "normalized_embeddings": True,
        "context_chunk_max_tokens": chunk_size,
        "context_chunk_overlap_tokens": chunk_overlap,
        "split": split,
        "row_count": row_count,
        "feature_dim": feature_dim,
        "creation_timestamp_utc": _utc_now(),
    }
    if extra:
        meta.update(extra)
    return meta


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_cache_metadata(
    cached_meta: Dict[str, Any],
    expected_meta: Dict[str, Any],
    cache_path: Path,
) -> None:
    """Raise ValueError if cached metadata is incompatible."""
    checks = [
        ("dataset_checksum", "Dataset checksum mismatch"),
        ("split_manifest_checksum", "Split manifest checksum mismatch"),
        ("encoder_name", "Encoder name mismatch"),
        ("embedding_dim", "Embedding dimension mismatch"),
        ("normalized_embeddings", "Normalization setting mismatch"),
        ("context_chunk_max_tokens", "Chunk size mismatch"),
        ("context_chunk_overlap_tokens", "Chunk overlap mismatch"),
        ("preprocessing_version", "Preprocessing version mismatch"),
        ("cache_format_version", "Cache format version mismatch"),
        ("feature_schema_version", "Feature schema version mismatch"),
    ]
    for key, msg in checks:
        if key not in expected_meta:
            continue
        if key not in cached_meta:
            raise ValueError(f"Missing required metadata key {key!r} in cache {cache_path}")
        if cached_meta[key] != expected_meta[key]:
            raise ValueError(
                f"{msg} for cache {cache_path}: "
                f"cached={cached_meta[key]}, expected={expected_meta[key]}"
            )


class EmbeddingCache:
    """Manages embedding and feature caches for a single split."""

    def __init__(self, cache_dir: Path, split: str) -> None:
        self._cache_dir = cache_dir
        self._split = split
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        """Directory containing this cache's artifacts."""
        return self._cache_dir

    def _embeddings_path(self, variant: str) -> Path:
        return self._cache_dir / f"embeddings_{self._split}_{variant}.npy"

    def _features_path(self) -> Path:
        return self._cache_dir / f"features_{self._split}.npy"

    def _metadata_path(self) -> Path:
        return self._cache_dir / f"cache_metadata_{self._split}.json"

    def _scaler_path(self) -> Path:
        return self._cache_dir / f"scaler_{self._split}.npz"

    def has_valid_cache(self, expected_meta: Dict[str, Any]) -> bool:
        """Check if a compatible cache exists."""
        meta_path = self._metadata_path()
        if not meta_path.exists():
            return False
        try:
            cached = json.loads(meta_path.read_text(encoding="utf-8"))
            _validate_cache_metadata(cached, expected_meta, meta_path)
            # Verify embeddings file exists
            for variant in ("q", "a", "c_a", "all_c"):
                if not self._embeddings_path(variant).exists():
                    return False
            return True
        except (ValueError, json.JSONDecodeError, KeyError):
            return False

    def load_metadata(self) -> Dict[str, Any]:
        meta_path = self._metadata_path()
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def load_embeddings(
        self, variant: str
    ) -> np.ndarray:
        path = self._embeddings_path(variant)
        return np.load(path, allow_pickle=False)

    def load_features(self) -> np.ndarray:
        return np.load(self._features_path(), allow_pickle=False)

    def save_embeddings(self, variant: str, data: np.ndarray) -> None:
        path = self._embeddings_path(variant)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, data, allow_pickle=False)
        tmp.replace(path)

    def save_features(self, features: np.ndarray) -> None:
        path = self._features_path()
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, features, allow_pickle=False)
        tmp.replace(path)

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        path = self._metadata_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def save_scaler(self, mean: np.ndarray, scale: np.ndarray) -> None:
        path = self._scaler_path()
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, mean=mean, scale=scale)
        tmp.replace(path)

    def load_scaler(self) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(self._scaler_path(), allow_pickle=False)
        return data["mean"], data["scale"]

    def cache_size_bytes(self) -> int:
        total = 0
        for p in self._cache_dir.iterdir():
            if p.is_file() and self._split in p.name:
                total += p.stat().st_size
        return total
