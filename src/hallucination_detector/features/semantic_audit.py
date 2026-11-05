"""Descriptive training-only similarity audit; no model selection or fitting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from hallucination_detector.features.embeddings import _compute_checksum
from hallucination_detector.features.preprocessing import PREPROCESSING_VERSION


def summarize_similarity(features: np.ndarray, labels: np.ndarray, index: int) -> Dict[str, Any]:
    """Summarize unscaled maximum answer-context cosine separately by label."""
    if features.ndim != 2 or features.shape[1] != 1543:
        raise ValueError("Expected a 1,543-dimensional primary feature matrix")
    if labels.shape != (len(features),) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected aligned binary labels containing both classes")
    scores = features[:, index]
    if not np.isfinite(scores).all():
        raise ValueError("Similarity scores must be finite")
    result = {}
    for name, label in (("grounded", 0), ("hallucinated", 1)):
        values = scores[labels == label]
        result[name] = {
            "label": label,
            "count": int(len(values)),
            "mean": float(values.mean()),
            "std_population": float(values.std()),
            "min": float(values.min()),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(values.max()),
        }
    return result


def run_train_semantic_audit(root: Path = Path(".")) -> Dict[str, Any]:
    """Read only training features/labels and save aggregate statistics and a figure."""
    import matplotlib.pyplot as plt

    train = root / "data/features/train"
    feature_path, label_path = train / "features.npy", train / "labels.npy"
    features = np.load(feature_path, allow_pickle=False)
    labels = np.load(label_path, allow_pickle=False)
    schema_path = root / "data/features/schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    index = next(s["dim_index"] for s in schema["scalar_features"]["order"]
                 if s["name"] == "max_answer_context_cosine")
    stats = summarize_similarity(features, labels, index)
    result = {
        "split": "train",
        "feature": "max_answer_context_cosine",
        "scaled": False,
        "preprocessing_version": PREPROCESSING_VERSION,
        "training_features_sha256": _compute_checksum(feature_path),
        "training_labels_sha256": _compute_checksum(label_path),
        "dataset_checksum": _compute_checksum(root / "data/raw/qa_data.json"),
        "split_manifest_checksum": _compute_checksum(root / "data/splits/halueval_original_split_manifest.json"),
        "statistics": stats,
        "interpretation": (
            "Descriptive training-only comparison. Separation and overlap may reflect "
            "grounding and benchmark shortcuts; these statistics do not establish "
            "generalization or causality and do not change the locked feature design."
        ),
    }
    output = root / "results/metrics/halueval_train_semantic_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-1, 1, 51)
    for name, label in (("Grounded", 0), ("Hallucinated", 1)):
        ax.hist(features[labels == label, index], bins=bins, alpha=0.55, label=name)
    ax.set(xlabel="Maximum answer-context chunk cosine (unscaled)",
           ylabel="Training examples", title="HaluEval QA: training-only semantic similarity")
    ax.legend()
    fig.tight_layout()
    figure = root / "results/figures/train_answer_context_cosine.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    return result
