"""Deterministic validation-only threshold and regularization selection."""
from __future__ import annotations

import inspect
import warnings
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from threadpoolctl import threadpool_limits


def validate_labels(labels: np.ndarray) -> None:
    """Require a nonempty binary vector with both locked classes."""
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Labels must contain both classes: 0=Grounded, 1=Hallucinated")


def validation_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Use label 1 as positive and continuous hallucination scores for AUC."""
    validate_labels(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Expected finite scores aligned with labels")
    predicted = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", pos_label=1, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def select_threshold(labels: np.ndarray, scores: np.ndarray, reference: float = 0.5) -> float:
    """Maximize F1 for score >= threshold; ties prefer reference proximity then lower value.

    Candidates are distinct observed scores, the reference, and nextafter(max, +inf)
    for the all-negative rule. Thus every achievable labeling is considered;
    proximity is defined over this finite candidate set, not a continuum.
    """
    validate_labels(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Expected finite scores aligned with labels")
    thresholds = np.unique(np.r_[scores, reference, np.nextafter(scores.max(), np.inf)])
    order = np.argsort(scores, kind="stable")
    prefix = np.r_[0, np.cumsum(labels[order])]
    positions = np.searchsorted(scores[order], thresholds, side="left")
    tp = labels.sum() - prefix[positions]
    predicted_positive = len(labels) - positions
    f1 = 2 * tp / (labels.sum() + predicted_positive)
    best = np.lexsort((thresholds, np.abs(thresholds - reference), -f1))[0]
    return float(thresholds[best])


def select_cosine(labels: np.ndarray, similarity: np.ndarray, directions: list) -> Dict[str, Any]:
    """Select direction and threshold on validation F1; direction ties prefer lower."""
    if set(directions) != {"lower", "higher"} or len(directions) != 2:
        raise ValueError("Both lower and higher cosine directions are required")
    candidates = []
    for direction in directions:
        sign = 1 if direction == "higher" else -1
        scores = sign * similarity
        threshold = select_threshold(labels, scores, reference=0.0)
        candidates.append({
            "direction": direction, "score_multiplier": sign,
            "score_threshold": threshold, "similarity_threshold": sign * threshold,
            "rule": "similarity >= threshold" if sign == 1 else "similarity <= threshold",
            "validation_metrics": validation_metrics(labels, scores, threshold),
        })
    selected = min(candidates, key=lambda r: (-r["validation_metrics"]["f1"], r["direction"] != "lower"))
    return {"selected": selected, "search": candidates}


def select_logistic(train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray,
                    validation_y: np.ndarray, config: Dict[str, Any]) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Fit every C on training only; choose thresholds and C using validation only."""
    validate_labels(train_y)
    validate_labels(validation_y)
    if (train_x.ndim != 2 or validation_x.ndim != 2 or
            train_x.shape[1] != validation_x.shape[1] or len(train_x) != len(train_y) or
            len(validation_x) != len(validation_y) or
            not np.isfinite(train_x).all() or not np.isfinite(validation_x).all()):
        raise ValueError("Training/validation matrices must be finite and aligned")
    rows, models = [], []
    for c in config["c_grid"]:
        params = dict(C=float(c), solver=config["solver"], max_iter=config["max_iter"],
                      tol=float(config["tol"]), random_state=config["random_state"], class_weight=None)
        # sklearn 1.8+ expresses pure L2 via l1_ratio=0; older versions use penalty.
        penalty = inspect.signature(LogisticRegression).parameters.get("penalty")
        if penalty is None or penalty.default == "deprecated":
            params["l1_ratio"] = 0.0
        else:
            params["penalty"] = "l2"
        model = LogisticRegression(**params)
        with threadpool_limits(limits=config["blas_threads"], user_api="blas"):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(train_x, train_y)
            probabilities = model.predict_proba(validation_x)[:, list(model.classes_).index(1)]
        threshold = select_threshold(validation_y, probabilities)
        rows.append({
            "C": float(c), "threshold": threshold,
            "validation_metrics": validation_metrics(validation_y, probabilities, threshold),
            "converged": not any(issubclass(w.category, ConvergenceWarning) for w in captured)
                         and int(model.n_iter_.max()) < config["max_iter"],
            "n_iter": model.n_iter_.astype(int).tolist(),
            "warnings": [str(w.message) for w in captured],
        })
        models.append(model)
    if not rows or not all(r["converged"] for r in rows):
        raise RuntimeError(f"All C candidates must converge before selection: {rows}")
    best = min(range(len(rows)), key=lambda i: (
        -rows[i]["validation_metrics"]["f1"], -rows[i]["validation_metrics"]["roc_auc"],
        rows[i]["C"], abs(rows[i]["threshold"] - 0.5), rows[i]["threshold"]))
    return models[best], {"selected": rows[best], "search": rows}
