"""Phase 5 orchestration over unchanged, Phase 4-verified feature artifacts."""
from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess  # Shared with source_commit; legacy callers patch this module.
from typing import Any

import numpy as np
import torch

from hallucination_detector.baselines.pipeline import (
    _json, _sha, load_development_split, schema_fingerprint,
)
from hallucination_detector.config import load_config
from hallucination_detector.reproduction import source_commit
from hallucination_detector.features.preprocessing import PREPROCESSING_VERSION
from hallucination_detector.features.scaler import load_scaler, transform_with_scaler
from hallucination_detector.features.schema import load_feature_schema
from hallucination_detector.mlp import train_seed, summarize_runs


def load_phase5_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check Phase 4 provenance and scaler identity without fitting or loading test data.

    Phase 4 verified the scaler against training moments. Matching its recorded
    checksums preserves that verification without fitting another scaler here.
    Only train/validation paths are constructed, even if other splits exist.
    """
    schema = _json(root / "configs/feature_schema.json")
    if schema != load_feature_schema() or schema != _json(root / "data/features/schema.json"):
        raise ValueError("Incompatible canonical feature schema")
    previous_path = root / "results/metrics/phase4_validation.json"
    previous = _json(previous_path)
    provenance = _json(root / "data/metadata/halueval_qa_provenance.json")
    manifest = root / "data/splits/halueval_original_split_manifest.json"
    anchor = previous["provenance"]
    if (previous["phase"] != 4 or previous["evaluation_split"] != "validation"
            or anchor["preprocessing_version"] != PREPROCESSING_VERSION
            or anchor["schema_fingerprint"] != schema_fingerprint(schema)
            or anchor["dataset_checksum"] != provenance["sha256"]
            or anchor["source_revision"] != provenance["source_revision"]
            or anchor["split_manifest_checksum"] != _sha(manifest)):
        raise ValueError("Incompatible Phase 4 schema, preprocessing, or provenance")
    expected_cache = {
        "cache_format_version": "1.0.0", "preprocessing_version": PREPROCESSING_VERSION,
        "feature_schema_version": schema["schema_version"],
        "dataset_checksum": provenance["sha256"], "split_manifest_checksum": _sha(manifest),
        "encoder_name": schema["encoder"]["model"], "embedding_dim": 384,
        "normalized_embeddings": True, "context_chunk_max_tokens": 192,
        "context_chunk_overlap_tokens": 32,
    }
    cache_metadata = {}
    for variant in ("questions", "answers", "contexts", "chunks"):
        cache = _json(root / f"data/embedding_cache/cache_metadata_{variant}.json")
        if any(cache.get(k) != v for k, v in expected_cache.items()):
            raise ValueError(f"Incompatible {variant} cache preprocessing or provenance")
        for key in ("encoder_revision", "sentence_transformers_version"):
            if cache.get(key) != anchor["encoder"].get(key):
                raise ValueError(f"Incompatible {variant} encoder provenance")
        cache_metadata[variant] = cache
    inputs = ["data/scaler/scaler.npz", "data/features/schema.json"]
    for split in ("train", "validation"):
        inputs.extend(f"data/features/{split}/{name}" for name in (
            "features.npy", "features_scaled.npy", "labels.npy", "example_ids.json", "source_ids.json"))
        inputs.append(f"data/processed/halueval_expanded_{split}.jsonl")
    checksums = {p: _sha(root / p) for p in inputs}
    if any(anchor["input_checksums"].get(p) != digest for p, digest in checksums.items()):
        raise ValueError("Feature/scaler provenance differs from verified Phase 4 inputs")
    splits = {s: {**load_development_split(root, s, schema["total_dim"]), "split": s}
              for s in ("train", "validation")}
    if set(splits["train"]["source_ids"]) & set(splits["validation"]["source_ids"]):
        raise ValueError("Source IDs overlap train and validation")
    mean, scale = load_scaler(root / "data/scaler/scaler.npz")
    if (mean.shape != (1543,) or scale.shape != (1543,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Incompatible training scaler")
    for data in splits.values():
        if not np.array_equal(transform_with_scaler(data["features"], mean, scale), data["features_scaled"]):
            raise ValueError("Saved scaled features are incompatible with the training scaler")
    return splits, {
        "schema_fingerprint": schema_fingerprint(schema), "primary_feature_dim": 1543,
        "preprocessing_version": PREPROCESSING_VERSION, "feature_schema_version": schema["schema_version"],
        "dataset_checksum": provenance["sha256"], "source_revision": provenance["source_revision"],
        "split_manifest_checksum": _sha(manifest), "input_checksums": checksums,
        "phase4_report_checksum": _sha(previous_path), "encoder_metadata": cache_metadata,
        "scaler_policy": "Reuse unchanged scaler verified against training moments in Phase 4; never fit in Phase 5",
    }


def validate_training_config(config: dict[str, Any]) -> None:
    """Fail instead of silently changing the locked research configuration."""
    expected = {"architecture": [1543, 256, 64, 1], "activation": "ReLU", "dropout": [0.2, 0.1],
                "loss": "BCEWithLogitsLoss", "optimizer": "AdamW", "learning_rate": 0.001,
                "weight_decay": 0.0001, "batch_size": 128, "max_epochs": 30, "early_stopping_patience": 5}
    if config["mlp"] != expected or config["mlp_seeds"] != [101, 202, 303]:
        raise ValueError("Incompatible locked MLP training configuration")
    if config["primary_features"]["total_dim"] != 1543 or config["labels"] != {"grounded": 0, "hallucinated": 1}:
        raise ValueError("Incompatible feature dimension or labels")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plot_histories(root: Path, runs: list[dict[str, Any]], figure_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(runs), figsize=(15, 4), sharey=True)
    for ax, run in zip(np.atleast_1d(axes), runs):
        history = _json(root / run["history_path"])
        epochs = [h["epoch"] for h in history]
        ax.plot(epochs, [h["training_loss"] for h in history], label="Training")
        ax.plot(epochs, [h["validation_loss"] for h in history], label="Validation")
        ax.axvline(run["best_epoch"], color="gray", linestyle="--", label="Best checkpoint")
        ax.set(title=f"Seed {run['seed']}", xlabel="Epoch", ylabel="Sample-weighted BCE loss")
        ax.legend()
    fig.suptitle("Phase 5: training and validation loss (no test evaluation)")
    fig.tight_layout()
    path = root / figure_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_phase5(root: Path = Path(".")) -> dict[str, Any]:
    """Run exactly three seeds on saved training-scaled features and stop at validation."""
    config = load_config()
    validate_training_config(config)
    settings, runtime = config["mlp"], config["phase5"]
    splits, compatibility = load_phase5_inputs(root)
    device = runtime["device"]
    torch.set_num_threads(runtime["torch_threads"])
    runs = []
    for seed in config["mlp_seeds"]:
        checkpoint = f"{runtime['checkpoint_dir']}/mlp_seed_{seed}.pt"
        print(f"Training MLP seed {seed} on {device}...", flush=True)
        run = train_seed(splits["train"], splits["validation"], seed, settings,
                         root / checkpoint, compatibility, device)
        history_path = f"{runtime['history_dir']}/seed_{seed}_history.json"
        write_json(root / history_path, run.pop("history"))
        run.update(history_path=history_path, checkpoint_path=checkpoint, checkpoint_checksum=_sha(root / checkpoint))
        runs.append(run)
        print(f"Seed {seed}: best epoch {run['best_epoch']}, validation F1={run['validation_metrics']['f1']:.6f}", flush=True)
    git_head = source_commit(root)
    report = {
        "phase": 5, "evaluation_split": "validation", "positive_class": 1,
        "confusion_matrix_order": "rows=true, columns=predicted; labels=[0,1]",
        "row_counts": {s: len(d["labels"]) for s, d in splits.items()},
        "configuration": settings, "seeds": config["mlp_seeds"], "runtime_configuration": runtime,
        "class_weight": None, "provenance": compatibility,
        "code_provenance": {"starting_commit": git_head, "uncommitted_phase5_implementation": True,
                            "source_checksums": {p: _sha(root / p) for p in (
                                "src/hallucination_detector/mlp.py", "src/hallucination_detector/mlp_pipeline.py",
                                "scripts/run_phase5.py", "configs/default.yaml")}},
        "versions": {"python": platform.python_version(), **{p: version(p) for p in (
            "torch", "numpy", "scikit-learn", "matplotlib", "sentence-transformers")}},
        "execution": {"device": device, "platform": platform.platform(),
                      "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                      "cudnn_deterministic": torch.backends.cudnn.deterministic,
                      "cudnn_benchmark": torch.backends.cudnn.benchmark,
                      "cuda_version": torch.version.cuda, "loader_workers": 0,
                      "shuffle": "training only; independent torch.Generator seeded per run",
                      "rngs_seeded": ["Python", "NumPy", "PyTorch", "DataLoader generator"],
                      "reproducibility_scope": "Same software/hardware and CPU settings; cross-platform bitwise identity is not guaranteed"},
        "selection_policy": {
            "checkpoint": "Strictly lowest sample-weighted mean validation BCE; exact ties keep earliest epoch",
            "early_stopping": "Stop after 5 consecutive epochs without strictly lower validation loss",
            "threshold": "After best checkpoint restoration: maximize validation hallucinated-class F1; score >= threshold; reuse Phase 4 distinct-score/reference/nextafter candidates, ties closest to 0.5 then smaller threshold",
            "representative_seed": "Median validation F1 across all three runs; exact F1 ties choose smaller seed",
            "fitting": "Training only; no refit, scaler fit, embedding regeneration, or test access",
        },
        "runs": runs, "summary": summarize_runs(runs), "loss_figure": runtime["figure_path"],
    }
    plot_histories(root, runs, runtime["figure_path"])
    write_json(root / runtime["report_path"], report)
    return report
