"""Training-only StandardScaler.

Fits a z-score scaler on training data only and transforms all splits
using the training-fitted parameters.  The scaler is saved as a NumPy
archive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def fit_scaler(train_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit StandardScaler parameters on training features only.

    Args:
        train_features: (N_train, D) feature matrix.

    Returns:
        (mean, scale) each of shape (D,).
    """
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    # Avoid division by zero: replace zero std with 1.0
    scale = np.where(std == 0.0, 1.0, std)
    return mean.astype(np.float32), scale.astype(np.float32)


def transform_with_scaler(
    features: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Apply a fitted scaler to a feature matrix.

    Args:
        features: (N, D) feature matrix.
        mean: (D,) training-fitted mean.
        scale: (D,) training-fitted scale.

    Returns:
        (N, D) scaled feature matrix.
    """
    scaled = (features - mean) / scale
    if not np.all(np.isfinite(scaled)):
        raise ValueError("Scaled features contain non-finite values")
    return scaled.astype(np.float32)


def save_scaler(path: Path, mean: np.ndarray, scale: np.ndarray) -> None:
    """Save scaler parameters atomically."""
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=str(path.parent))
    try:
        import os
        os.close(fd)
        np.savez(tmp_path, mean=mean, scale=scale)
        import shutil
        shutil.move(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_scaler(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load scaler parameters."""
    data = np.load(path, allow_pickle=False)
    return data["mean"], data["scale"]
