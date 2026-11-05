"""Locked Phase 6 inference, compatibility checks, and ordered result persistence."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from hallucination_detector.baselines.control import answer_surface_features
from hallucination_detector.baselines.pipeline import _json, _sha, load_models, schema_fingerprint
from hallucination_detector.baselines.selection import validate_labels, validation_metrics
from hallucination_detector.config import load_config
from hallucination_detector.data.split_phase2 import _context_group_id, deterministic_example_id
from hallucination_detector.data.validate import compute_source_id
from hallucination_detector.features.scaler import load_scaler, transform_with_scaler
from hallucination_detector.mlp import load_checkpoint
from hallucination_detector.mlp_pipeline import load_phase5_inputs, write_json

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
LOCK_PATH = "results/metrics/phase6/evaluation_manifest.json"
SMOKE_PATH = "results/metrics/phase6/development_smoke.json"


def load_fixed_models(root: Path, phase4: dict, phase5: dict) -> dict:
    """Restore trusted local models and require agreement with validation reports."""
    config = load_config()
    schema = _json(root / "configs/feature_schema.json")
    baseline_path = root / config["baselines"]["checkpoint_dir"] / "baselines.joblib"
    baseline = load_models(baseline_path, schema)
    mean, scale = load_scaler(root / "data/scaler/scaler.npz")
    if (baseline["primary_schema"] != schema or baseline["provenance"] != phase4["provenance"]
            or not np.array_equal(baseline["primary_scaler"]["mean"], mean)
            or not np.array_equal(baseline["primary_scaler"]["scale"], scale)):
        raise ValueError("Baseline schema/scaler/provenance incompatibility")
    for key, model_key, threshold_key, dimension in (
        ("primary_logistic", "primary_model", "primary_threshold", 1543),
        ("negative_control", "control_model", "control_threshold", 6),
    ):
        selected, model = phase4[key]["selected"], baseline[model_key]
        if (model.n_features_in_ != dimension or list(model.classes_) != [0, 1]
                or model.C != selected["C"] or baseline[threshold_key] != selected["threshold"]):
            raise ValueError("Baseline model or fixed threshold disagrees with report")
    control = phase4["negative_control"]["scaler"]
    if any(not np.array_equal(getattr(baseline["control_scaler"], attr), control[key])
           for attr, key in (("mean_", "mean"), ("scale_", "scale"), ("var_", "var"))):
        raise ValueError("Negative-control scaler disagrees with report")
    if baseline["cosine"] != phase4["cosine"]["selected"]:
        raise ValueError("Cosine rule disagrees with report")
    if (phase5["seeds"] != config["mlp_seeds"]
            or [r["seed"] for r in phase5["runs"]] != config["mlp_seeds"]
            or phase5["summary"]["representative_seed"] != 202):
        raise ValueError("Expected all three seeds and validation-selected representative 202")
    mlps = {}
    for run in phase5["runs"]:
        path = root / run["checkpoint_path"]
        if _sha(path) != run["checkpoint_checksum"]:
            raise ValueError("MLP checkpoint checksum mismatch")
        model, checkpoint = load_checkpoint(path, phase5["provenance"])
        if (checkpoint["configuration"] != phase5["configuration"]
                or checkpoint["configuration"] != config["mlp"]
                or any(checkpoint[k] != run[k] for k in (
                    "seed", "threshold", "best_epoch", "best_validation_loss"))):
            raise ValueError("MLP checkpoint decisions disagree with validation report")
        mlps[run["seed"]] = model
    return {"baseline": baseline, "mlps": mlps, "mean": mean, "scale": scale}


def predict_mlp(model: torch.nn.Module, features: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """Dedicated ordered final inference; Phase 5 guards remain unchanged."""
    if (features.ndim != 2 or features.shape[1] != 1543 or len(features) == 0
            or not np.isfinite(features).all()):
        raise ValueError("Expected finite nonempty 1543-dimensional evaluation features")
    model.eval()
    with torch.inference_mode():
        return np.concatenate([
            torch.sigmoid(model(torch.as_tensor(features[i:i+batch_size], dtype=torch.float32))).cpu().numpy()
            for i in range(0, len(features), batch_size)
        ])


def score_models(data: dict, models: dict, decisions: dict, *, sanity: bool = False) -> list[dict]:
    """Score unchanged models; no training, selection, or scaler fitting."""
    baseline = models["baseline"]
    cosine = decisions["cosine"]
    scores = [("cosine", None, cosine["score_multiplier"] * data["features"][:, 1536],
               cosine["score_threshold"])]
    primary = baseline["primary_model"]
    scores.append(("primary_logistic", None, primary.predict_proba(data["features_scaled"])[:, 1],
                   decisions["primary_logistic"]["threshold"]))
    for run in decisions["mlp_runs"]:
        if sanity and run["seed"] != decisions["representative_mlp_seed"]:
            continue
        scores.append(("mlp", run["seed"], predict_mlp(models["mlps"][run["seed"]], data["features_scaled"]),
                       run["threshold"]))
    if not sanity:
        control_x = baseline["control_scaler"].transform(answer_surface_features(data["answers"]))
        scores.append(("negative_control", None, baseline["control_model"].predict_proba(control_x)[:, 1],
                       decisions["negative_control"]["threshold"]))
    return [{"model": model, "seed": seed, "threshold": threshold, "scores": values,
             "metrics": validation_metrics(data["labels"], values, threshold)}
            for model, seed, values, threshold in scores]


def aggregate_mlp(results: list[dict]) -> dict:
    """Aggregate all three fixed runs without choosing a representative from test."""
    runs = [r for r in results if r["model"] == "mlp"]
    if sorted(r["seed"] for r in runs) != [101, 202, 303]:
        raise ValueError("Three distinct core MLP seeds required")
    return {"seeds": [101, 202, 303], "ddof": 1,
            "metrics": {k: {"mean": np.mean([r["metrics"][k] for r in runs], axis=0).tolist(),
                            "sample_sd": np.std([r["metrics"][k] for r in runs], axis=0, ddof=1).tolist()}
                        for k in (*METRICS, "confusion_matrix")}}


def persist_predictions(path: Path, data: dict, results: list[dict]) -> None:
    """Write text-free, model-major predictions preserving identical example order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["example_id", "source_id", "true_label", "continuous_score", "predicted_label", "model", "seed"])
        for result in results:
            scores = result["scores"]
            if len(scores) != len(data["labels"]):
                raise ValueError("Prediction alignment mismatch")
            for eid, sid, label, score in zip(data["example_ids"], data["source_ids"], data["labels"], scores):
                writer.writerow([eid, sid, int(label), float(score), int(score >= result["threshold"]),
                                 result["model"], result["seed"]])


def prepare_evaluation(root: Path) -> dict:
    """Save a lock and reproduce validation metrics without opening test inputs."""
    if (root / LOCK_PATH).exists():
        raise FileExistsError("Evaluation manifest already exists; preserve the existing lock")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    splits, compatibility = load_phase5_inputs(root)
    phase4 = _json(root / "results/metrics/phase4_validation.json")
    phase5 = _json(root / "results/metrics/phase5_validation.json")
    if phase5["provenance"] != compatibility or phase5["evaluation_split"] != "validation":
        raise ValueError("Phase 5 provenance disagrees with verified development artifacts")
    models = load_fixed_models(root, phase4, phase5)
    decisions = {
        "cosine": phase4["cosine"]["selected"],
        "primary_logistic": phase4["primary_logistic"]["selected"],
        "negative_control": phase4["negative_control"]["selected"],
        "mlp_runs": phase5["runs"], "representative_mlp_seed": phase5["summary"]["representative_seed"],
        "shuffled_context_seed": load_config()["split"]["seed"],
    }
    if decisions["shuffled_context_seed"] != 42:
        raise ValueError("Prespecified shuffled-context seed is 42")
    paths = set(compatibility["input_checksums"])
    paths.update(["configs/default.yaml", "configs/feature_schema.json", "data/metadata/halueval_qa_provenance.json",
                  "data/splits/halueval_original_split_manifest.json", "results/metrics/phase4_validation.json",
                  "results/metrics/phase5_validation.json", "checkpoints/phase4/baselines.joblib"])
    paths.update(r["checkpoint_path"] for r in phase5["runs"])
    paths.update(p.relative_to(root).as_posix() for p in (root / "src/hallucination_detector").rglob("*.py"))
    paths.update(p.relative_to(root).as_posix() for p in (root / "tests").glob("test*.py"))
    paths.add("scripts/run_phase6.py")
    paths.update(f"data/embedding_cache/cache_metadata_{v}.json" for v in ("questions", "answers", "contexts", "chunks"))
    manifest = {"phase": 6, "status": "locked before test input access", "compatibility": compatibility,
                "decisions": decisions, "input_checksums": {p: _sha(root / p) for p in sorted(paths)},
                "metric_definitions": {"positive_class": "1=Hallucinated; 0=Grounded", "prediction": "score >= fixed threshold",
                    "accuracy": "(TN+TP)/N", "precision": "TP/(TP+FP)", "recall": "TP/(TP+FN)",
                    "f1": "2TP/(2TP+FP+FN); primary metric", "zero_division": 0,
                    "roc_auc": "sklearn ROC-AUC from continuous hallucination scores; never predicted labels",
                    "confusion_matrix": "[[TN, FP], [FN, TP]]", "mlp_sd_ddof": 1},
                "test_policy": "One planned evaluation; no fitting, tuning, checkpoint/seed/threshold selection",
                "shuffle_algorithm": "Seeded group order and within-group order; circular shift by largest duplicate-context group size; fail if infeasible; no result-based shuffle selection"}
    write_json(root / LOCK_PATH, manifest)
    with threadpool_limits(limits=1, user_api="blas"):
        results = score_models(splits["validation"], models, decisions)
    for result in results:
        expected = (next(r["validation_metrics"] for r in phase5["runs"] if r["seed"] == result["seed"])
                    if result["model"] == "mlp" else
                    phase4[result["model"]]["selected"]["validation_metrics"])
        if result["metrics"] != expected:
            raise ValueError(f"Development smoke differs from saved validation metrics: {result['model']}")
    smoke = {"evaluation_manifest_checksum": _sha(root / LOCK_PATH), "split": "validation",
             "all_saved_validation_metrics_reproduced": True, "test_inputs_opened": False,
             "results": [{k: v for k, v in r.items() if k != "scores"} for r in results]}
    write_json(root / SMOKE_PATH, smoke)
    return smoke


def verify_lock(root: Path) -> dict:
    """Verify the persisted lock and development smoke before any final data access."""
    manifest = _json(root / LOCK_PATH)
    smoke = _json(root / SMOKE_PATH)
    if (smoke["evaluation_manifest_checksum"] != _sha(root / LOCK_PATH)
            or not smoke["all_saved_validation_metrics_reproduced"] or smoke["test_inputs_opened"]):
        raise ValueError("A successful pre-test development smoke check is required")
    for path, digest in manifest["input_checksums"].items():
        if _sha(root / path) != digest:
            raise ValueError(f"Evaluation lock input changed: {path}")
    return manifest


def validate_evaluation_alignment(rows: list[dict], ids: list[str], sources: list[str], labels: np.ndarray,
                                  manifest: list[dict]) -> None:
    """Verify Phase 2 source/variant identities and labels against the split manifest."""
    validate_labels(labels)
    if (ids != [r["example_id"] for r in rows] or sources != [r["source_id"] for r in rows]
            or labels.tolist() != [r["label"] for r in rows] or len(ids) != len(set(ids))):
        raise ValueError("Evaluation example/label alignment mismatch")
    by_source = defaultdict(list)
    for row in rows:
        for field in ("question", "candidate_answer", "reference_context"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError("Invalid evaluation text")
        by_source[row["source_id"]].append(row)
    source_manifest = {r["source_id"]: r for r in manifest if r["split"] == "test"}
    if set(by_source) != set(source_manifest):
        raise ValueError("Test sources disagree with split manifest")
    groups = defaultdict(set)
    for row in manifest:
        groups[row["duplicate_context_group_id"]].add(row["split"])
    if len({r["source_id"] for r in manifest}) != len(manifest) or any(len(v) != 1 for v in groups.values()):
        raise ValueError("Source or duplicate-context leakage in manifest")
    for sid, pair in by_source.items():
        if len(pair) != 2 or sorted(r["label"] for r in pair) != [0, 1]:
            raise ValueError("Exactly two balanced paired examples required")
        pair = sorted(pair, key=lambda r: r["label"])
        for row, variant, name in zip(pair, ("right_answer", "hallucinated_answer"), ("Grounded", "Hallucinated")):
            if (row["split"] != "test" or row["answer_variant"] != variant or row["label_name"] != name
                    or row["example_id"] != deterministic_example_id(sid, variant)
                    or _context_group_id(row["reference_context"]) != source_manifest[sid]["duplicate_context_group_id"]):
                raise ValueError("Test variant, label, ID, or context provenance mismatch")
        if any(pair[0][key] != pair[1][key] for key in ("question", "reference_context")):
            raise ValueError("Paired inputs disagree")
        original = {"question": pair[0]["question"], "knowledge": pair[0]["reference_context"],
                    "right_answer": pair[0]["candidate_answer"], "hallucinated_answer": pair[1]["candidate_answer"]}
        if compute_source_id(original) != sid:
            raise ValueError("Processed source text does not match stable source identity")


def load_evaluation_split(root: Path, manifest: dict) -> dict:
    """Dedicated test loader; require a verified lock before opening test payloads."""
    if verify_lock(root) != manifest:
        raise ValueError("Unexpected evaluation manifest")
    schema = _json(root / "data/features/schema.json")
    if schema_fingerprint(schema) != manifest["compatibility"]["schema_fingerprint"]:
        raise ValueError("Test feature schema mismatch")
    path = root / "data/features/test"
    data = {k: np.load(path / f"{k}.npy", allow_pickle=False) for k in ("features", "features_scaled", "labels")}
    data.update(example_ids=_json(path / "example_ids.json"), source_ids=_json(path / "source_ids.json"))
    rows = [json.loads(line) for line in (root / "data/processed/halueval_expanded_test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_evaluation_alignment(rows, data["example_ids"], data["source_ids"], data["labels"],
                                  _json(root / "data/splits/halueval_original_split_manifest.json"))
    n = len(rows)
    if any(data[k].shape != (n, schema["total_dim"]) or not np.isfinite(data[k]).all()
           for k in ("features", "features_scaled")):
        raise ValueError("Invalid test feature dimensions or values")
    meta = _json(path / "metadata.json")
    if (meta["split"] != "test" or meta["feature_dim"] != 1543 or meta["row_count"] != n
            or meta["label_counts"] != {"grounded": n // 2, "hallucinated": n // 2}):
        raise ValueError("Test class counts/metadata mismatch")
    mean, scale = load_scaler(root / "data/scaler/scaler.npz")
    if not np.array_equal(transform_with_scaler(data["features"], mean, scale), data["features_scaled"]):
        raise ValueError("Test features do not reuse the locked training scaler")
    data.update(rows=rows, answers=[r["candidate_answer"] for r in rows])
    return data
