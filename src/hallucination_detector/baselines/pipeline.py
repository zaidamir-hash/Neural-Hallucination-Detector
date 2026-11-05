"""Phase 4 workflow restricted to training and validation inputs."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from hallucination_detector.config import load_config
from hallucination_detector.features.preprocessing import PREPROCESSING_VERSION
from hallucination_detector.features.scaler import load_scaler, fit_scaler, transform_with_scaler
from hallucination_detector.baselines.control import CONTROL_SCHEMA, answer_surface_features
from hallucination_detector.baselines.selection import select_cosine, select_logistic, validate_labels


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schema_fingerprint(schema: Dict[str, Any]) -> str:
    """Hash schema contents independently of JSON formatting and line endings."""
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()


def load_development_split(root: Path, split: str, dimension: int) -> Dict[str, Any]:
    """Refuse all splits other than train/validation before any file access."""
    if split not in ("train", "validation"):
        raise ValueError("Phase 4 accepts only train and validation; test inputs are forbidden")
    path = root / "data/features" / split
    arrays = {name: np.load(path / f"{name}.npy", allow_pickle=False)
              for name in ("features", "features_scaled", "labels")}
    raw, scaled, labels = (arrays[n] for n in ("features", "features_scaled", "labels"))
    validate_labels(labels)
    if raw.shape != (len(labels), dimension) or scaled.shape != raw.shape:
        raise ValueError(f"{split}: expected aligned {dimension}-dimensional features")
    if not np.isfinite(raw).all() or not np.isfinite(scaled).all():
        raise ValueError(f"{split}: non-finite features")
    if int((labels == 0).sum()) != int((labels == 1).sum()):
        raise ValueError(f"{split}: unexpected imbalance; review class-weight policy")
    rows_path = root / "data/processed" / f"halueval_expanded_{split}.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = _json(path / "example_ids.json")
    sources = _json(path / "source_ids.json")
    if (ids != [r["example_id"] for r in rows] or len(set(ids)) != len(ids) or
            sources != [r["source_id"] for r in rows] or
            labels.tolist() != [r["label"] for r in rows] or
            any(r["split"] != split for r in rows)):
        raise ValueError(f"{split}: processed rows do not align with feature IDs/labels")
    arrays["answers"] = [r["candidate_answer"] for r in rows]
    arrays["source_ids"] = sources
    return arrays


def save_models(path: Path, bundle: Dict[str, Any]) -> None:
    """Save fitted estimators and parameters in an ignored checkpoint directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_models(path: Path, expected_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Load a trusted local checkpoint and reject incompatible feature schemas."""
    bundle = joblib.load(path)
    if (bundle["schema_fingerprint"] != schema_fingerprint(expected_schema) or
            bundle["preprocessing_version"] != PREPROCESSING_VERSION or
            bundle["control_schema"] != CONTROL_SCHEMA):
        raise ValueError("Incompatible feature schema or preprocessing in Phase 4 checkpoint")
    return bundle


def run_phase4(root: Path = Path(".")) -> Dict[str, Any]:
    """Fit Phase 4 models and save validation-only results; never open test features."""
    config = load_config()
    settings = config["baselines"]
    schema = _json(root / "configs/feature_schema.json")
    if schema != _json(root / "data/features/schema.json") or schema["total_dim"] != 1543:
        raise ValueError("Phase 3 feature schema does not match the canonical schema")
    dimension = config["primary_features"]["total_dim"]
    if dimension != schema["total_dim"]:
        raise ValueError("Canonical feature dimension mismatch")
    cosine_index = next(f["dim_index"] for f in schema["scalar_features"]["order"]
                        if f["name"] == "max_answer_context_cosine")
    if cosine_index != 1536:
        raise ValueError("Cosine feature must be column 1536")
    provenance = _json(root / "data/metadata/halueval_qa_provenance.json")
    manifest_sha = _sha(root / "data/splits/halueval_original_split_manifest.json")
    cache_meta = _json(root / "data/embedding_cache/cache_metadata_questions.json")
    if (cache_meta["preprocessing_version"] != PREPROCESSING_VERSION or
            cache_meta["feature_schema_version"] != schema["schema_version"] or
            cache_meta["dataset_checksum"] != provenance["sha256"] or
            cache_meta["split_manifest_checksum"] != manifest_sha):
        raise ValueError("Phase 3 cache provenance is incompatible")
    splits = {s: load_development_split(root, s, dimension) for s in ("train", "validation")}
    train, validation = splits["train"], splits["validation"]
    if set(train["source_ids"]) & set(validation["source_ids"]):
        raise ValueError("Source IDs overlap training and validation")
    mean, scale = load_scaler(root / "data/scaler/scaler.npz")
    expected_mean, expected_scale = fit_scaler(train["features"])
    if not np.array_equal(mean, expected_mean) or not np.array_equal(scale, expected_scale):
        raise ValueError("Primary scaler is not the training-only Phase 3 scaler")
    for split in splits.values():
        if not np.array_equal(transform_with_scaler(split["features"], mean, scale), split["features_scaled"]):
            raise ValueError("Saved scaled features do not match training scaler")
    cosine = select_cosine(validation["labels"], validation["features"][:, cosine_index],
                           settings["cosine_directions"])
    primary_model, primary = select_logistic(train["features_scaled"], train["labels"],
                                             validation["features_scaled"], validation["labels"], settings)
    train_control = answer_surface_features(train["answers"])
    validation_control = answer_surface_features(validation["answers"])
    control_scaler = StandardScaler().fit(train_control)
    control_model, control = select_logistic(control_scaler.transform(train_control), train["labels"],
                                             control_scaler.transform(validation_control), validation["labels"], settings)
    inputs = [root / "data/scaler/scaler.npz", root / "data/features/schema.json"]
    for split in splits:
        inputs.extend((root / "data/features" / split).glob("*.npy"))
        inputs.extend((root / "data/features" / split).glob("*_ids.json"))
        inputs.append(root / "data/processed" / f"halueval_expanded_{split}.jsonl")
    report = {
        "phase": 4, "evaluation_split": "validation", "positive_class": 1,
        "confusion_matrix_order": "rows=true, columns=predicted; labels=[0,1]",
        "row_counts": {s: len(data["labels"]) for s, data in splits.items()},
        "label_counts": {s: {str(label): int((data["labels"] == label).sum()) for label in (0, 1)}
                         for s, data in splits.items()},
        "primary_feature_dim": dimension,
        "configuration": settings,
        "selection_policy": {
            "threshold": "score >= threshold; candidates=distinct scores, reference, nextafter(max,+inf); highest F1, closest reference, smaller threshold",
            "probability_reference": 0.5, "cosine_score_reference": 0.0,
            "cosine_direction": "highest validation F1; exact ties prefer lower similarity means hallucinated",
            "logistic": "highest validation F1, highest validation ROC-AUC, smaller C, threshold closest to 0.5, smaller threshold",
            "fitting": "training only; no refit on combined train/validation",
            "cosine_authorization": "User explicitly authorized validation-selected direction in Phase 4 after the Phase 3 audit; supersedes original expected direction, not the feature design",
        },
        "provenance": {
            "dataset_checksum": provenance["sha256"], "source_revision": provenance["source_revision"],
            "split_manifest_checksum": manifest_sha, "phase3_commit": settings["phase3_commit"],
            "preprocessing_version": PREPROCESSING_VERSION, "feature_schema_version": schema["schema_version"],
            "schema_fingerprint": schema_fingerprint(schema), "encoder": cache_meta,
            "input_checksums": {p.relative_to(root).as_posix(): _sha(p) for p in sorted(inputs)},
        },
        "versions": {"python": platform.python_version(), **{p: version(p) for p in (
            "numpy", "scikit-learn", "scipy", "joblib", "threadpoolctl", "sentence-transformers", "torch")}},
        "model_specification": {"type": "sklearn.linear_model.LogisticRegression", "penalty": "L2",
                                "class_weight": None, "fit_intercept": True},
        "cosine": cosine, "primary_logistic": primary,
        "negative_control": {
            **control, "role": "answer-only style/length negative control, not a primary model",
            "feature_schema": CONTROL_SCHEMA,
            "scaler": {"mean": control_scaler.mean_.tolist(), "scale": control_scaler.scale_.tolist(),
                       "var": control_scaler.var_.tolist(), "n_samples_seen": int(control_scaler.n_samples_seen_)},
        },
    }
    bundle = {
        "schema_fingerprint": schema_fingerprint(schema), "primary_schema": schema,
        "preprocessing_version": PREPROCESSING_VERSION, "control_schema": CONTROL_SCHEMA,
        "primary_model": primary_model, "control_model": control_model,
        "control_scaler": control_scaler, "primary_scaler": {"mean": mean, "scale": scale},
        "cosine": cosine["selected"], "primary_threshold": primary["selected"]["threshold"],
        "control_threshold": control["selected"]["threshold"],
        "provenance": report["provenance"], "versions": report["versions"],
    }
    save_models(root / settings["checkpoint_dir"] / "baselines.joblib", bundle)
    output = root / settings["report_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
