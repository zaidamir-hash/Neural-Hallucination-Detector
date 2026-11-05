"""One planned final test evaluation followed by the prespecified context sanity test."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from hallucination_detector.baselines.pipeline import _json, _sha
from hallucination_detector.context_sanity import (
    apply_context_mapping, context_mapping, load_frozen_cache, recompute_context_features,
)
from hallucination_detector.evaluation import (
    LOCK_PATH, METRICS, aggregate_mlp, load_evaluation_split, load_fixed_models,
    persist_predictions, score_models, verify_lock,
)
from hallucination_detector.features.scaler import transform_with_scaler
from hallucination_detector.mlp_pipeline import write_json

OUTPUT = "results/metrics/phase6"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def result_label(result: dict) -> str:
    if result["model"] == "mlp":
        return f"MLP seed {result['seed']}" + (" (representative)" if result["seed"] == 202 else "")
    return {"cosine": "Cosine", "primary_logistic": "Primary Logistic Regression",
            "negative_control": "Answer-only negative control (separate diagnostic)"}[result["model"]]


def without_scores(results: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "scores"} for r in results]


def comparison_table(results: list[dict], summary: dict) -> str:
    lines = ["| Approach (final test) | Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC-AUC |",
             "|---|---:|---:|---:|---:|---:|"]
    for result in results:
        if result["model"] == "negative_control":
            continue
        lines.append("| " + result_label(result) + " | " + " | ".join(f"{result['metrics'][k]:.6f}" for k in METRICS) + " |")
    for name, field in (("MLP three-seed mean", "mean"), ("MLP sample SD (ddof=1)", "sample_sd")):
        lines.append("| " + name + " | " + " | ".join(f"{summary['metrics'][k][field]:.6f}" for k in METRICS) + " |")
    control = next(r for r in results if r["model"] == "negative_control")
    lines.extend(["", "Separate answer-only style/length diagnostic:", "",
                  "| Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC-AUC |",
                  "|---:|---:|---:|---:|---:|",
                  "| " + " | ".join(f"{control['metrics'][k]:.6f}" for k in METRICS) + " |", ""])
    return "\n".join(lines)


def plot_evaluation(root: Path, labels: np.ndarray, results: list[dict]) -> list[str]:
    """All run confusion matrices; representative 202 only for MLP ROC/PR curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    directory = root / "results/figures/phase6"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, result in zip(axes.ravel(), results):
        cm = np.asarray(result["metrics"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.sum(axis=1)))
        for (i, j), count in np.ndenumerate(cm):
            ax.text(j, i, str(count), ha="center", va="center", color="white" if count > cm.max()/2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Grounded", "Hallucinated"],
               yticklabels=["Grounded", "Hallucinated"], xlabel="Predicted", ylabel="True",
               title=result_label(result).replace(" (separate diagnostic)", "\n(separate diagnostic)"))
    fig.suptitle("Locked final test confusion matrices: [[TN, FP], [FN, TP]]")
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    paths.append("results/figures/phase6/confusion_matrices.png")
    selected = [r for r in results if r["model"] != "mlp" or r["seed"] == 202]
    for kind in ("roc", "precision_recall"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        for result in selected:
            ax = axes[int(result["model"] == "negative_control")]
            if kind == "roc":
                x, y, _ = roc_curve(labels, result["scores"], pos_label=1)
                label = f"{result_label(result)}; AUC={result['metrics']['roc_auc']:.4f}"
            else:
                y, x, _ = precision_recall_curve(labels, result["scores"], pos_label=1)
                label = result_label(result)
            ax.plot(x, y, label=label)
        for ax, title in zip(axes, ("Three primary approaches (MLP seed 202)", "Separate answer-only negative control")):
            if kind == "roc":
                ax.plot([0, 1], [0, 1], "--", color="gray")
            else:
                ax.axhline(float(labels.mean()), linestyle="--", color="gray")
            ax.set(xlabel="False positive rate" if kind == "roc" else "Recall (Hallucinated)",
                   ylabel="True positive rate" if kind == "roc" else "Precision (Hallucinated)",
                   xlim=(0, 1), ylim=(0, 1.02), title=title)
            ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=8)
        fig.suptitle("Locked final test: " + ("ROC" if kind == "roc" else "precision-recall"))
        fig.tight_layout()
        fig.savefig(directory / f"{kind}.png", dpi=150)
        plt.close(fig)
        paths.append(f"results/figures/phase6/{kind}.png")
    return paths


def run_final_evaluation(root: Path) -> dict:
    """Persist primary outputs before shuffling; refuse to overwrite any prior attempt."""
    manifest = verify_lock(root)
    output = root / OUTPUT
    attempt = output / "execution.json"
    if attempt.exists() or (output / "primary_metrics.json").exists():
        raise FileExistsError("Final evaluation already attempted; preserve outputs and document any defect before reevaluation")
    models = load_fixed_models(root, _json(root / "results/metrics/phase4_validation.json"),
                               _json(root / "results/metrics/phase5_validation.json"))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    execution = {"status": "started", "started_utc": _now(), "planned_evaluation_number": 1, "reruns": [],
                 "evaluation_manifest_checksum": _sha(root / LOCK_PATH),
                 "starting_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                                   text=True, check=True).stdout.strip(),
                 "versions": {"python": platform.python_version(), **{p: version(p) for p in (
                     "numpy", "torch", "scikit-learn", "matplotlib", "sentence-transformers")}},
                 "runtime": {"device": "cpu", "torch_threads": 1, "blas_threads": 1,
                             "deterministic_algorithms": True}}
    write_json(attempt, execution)
    try:
        data = load_evaluation_split(root, manifest)
        cache = load_frozen_cache(root, manifest["compatibility"]["encoder_metadata"])
        reconstructed = recompute_context_features(data["rows"], cache)
        if not np.array_equal(reconstructed, data["features"]):
            raise ValueError("Test features differ from Phase 3 reconstruction using frozen compatible caches")
        input_paths = list((root / "data/features/test").glob("*"))
        input_paths.append(root / "data/processed/halueval_expanded_test.jsonl")
        input_paths.extend(p for p in (root / "data/embedding_cache").glob("*") if p.is_file())
        input_checksums = {p.relative_to(root).as_posix(): _sha(p) for p in sorted(input_paths)}
        with threadpool_limits(limits=1, user_api="blas"):
            results = score_models(data, models, manifest["decisions"])
        summary = aggregate_mlp(results)
        primary = {"phase": 6, "split": "test", "evaluation_manifest_checksum": _sha(root / LOCK_PATH),
                   "row_count": len(data["labels"]), "source_count": len(set(data["source_ids"])),
                   "label_counts": {str(k): int((data["labels"] == k).sum()) for k in (0, 1)},
                   "input_checksums": input_checksums, "feature_reconstruction_exact": True,
                   "results": without_scores(results), "mlp_three_seed_summary": summary,
                   "representative_mlp_seed": 202, "metric_definitions": manifest["metric_definitions"],
                   "negative_control_role": "Separate answer-only diagnostic, not a primary model",
                   "qualitative_handoff": {"model": "mlp", "seed": 202, "predictions": f"{OUTPUT}/primary_predictions.csv",
                                           "analysis_performed": False}}
        persist_predictions(output / "primary_predictions.csv", data, results)
        primary["figures"] = plot_evaluation(root, data["labels"], results)
        (output / "comparison.md").write_text(comparison_table(results, summary), encoding="utf-8")
        write_json(output / "primary_metrics.json", primary)
        primary_paths = [output / p for p in ("primary_metrics.json", "primary_predictions.csv", "comparison.md")]
        primary_paths.extend(root / p for p in primary["figures"])
        saved = {"primary_saved_utc": _now(), "before_shuffled_context_analysis": True,
                 "output_checksums": {p.relative_to(root).as_posix(): _sha(p) for p in primary_paths}}
        write_json(output / "primary_saved.json", saved)
        print("Primary test metrics, predictions, comparison and plots saved; starting fixed context shuffle.", flush=True)

        mapping, integrity = context_mapping(data["rows"], manifest["decisions"]["shuffled_context_seed"])
        shuffled_rows = apply_context_mapping(data["rows"], mapping)
        shuffled_raw = recompute_context_features(shuffled_rows, cache)
        # These blocks/scalars depend on question/answer only and must be bitwise unchanged.
        invariant_indices = np.r_[np.arange(768, 1536), 1538, 1542]
        if not np.array_equal(shuffled_raw[:, invariant_indices], data["features"][:, invariant_indices]):
            raise ValueError("Shuffling changed question/answer-only interactions")
        integrity["checks"].update(
            paired_replacement_contexts=all(len({r["reference_context"] for r in shuffled_rows if r["source_id"] == sid}) == 1 for sid in mapping),
            question_answer_inputs_and_order_unchanged=all(
                all(r[k] == s[k] for k in ("question", "candidate_answer", "example_id", "source_id", "label"))
                for r, s in zip(data["rows"], shuffled_rows)),
            question_answer_features_unchanged=True,
            training_scaler_unchanged=_sha(root / "data/scaler/scaler.npz") == manifest["input_checksums"]["data/scaler/scaler.npz"],
            original_features_reconstructed_exactly=True,
        )
        if not all(integrity["checks"].values()):
            raise ValueError("Shuffled-context integrity failure")
        write_json(output / "shuffled_context_mapping.json", {**integrity,
                   "source_to_donor": [{"source_id": s, "donor_source_id": mapping[s]} for s in sorted(mapping)]})
        shuffled = {**data, "features": shuffled_raw,
                    "features_scaled": transform_with_scaler(shuffled_raw, models["mean"], models["scale"])}
        with threadpool_limits(limits=1, user_api="blas"):
            shuffled_results = score_models(shuffled, models, manifest["decisions"], sanity=True)
        comparisons = []
        for result in shuffled_results:
            original = next(r for r in results if (r["model"], r["seed"]) == (result["model"], result["seed"]))
            comparisons.append({"model": result["model"], "seed": result["seed"], "threshold": result["threshold"],
                                "original": original["metrics"], "shuffled": result["metrics"],
                                "delta_shuffled_minus_original": {
                                    k: (np.asarray(result["metrics"][k]) - np.asarray(original["metrics"][k])).tolist()
                                    for k in (*METRICS, "confusion_matrix")}})
        persist_predictions(output / "shuffled_predictions.csv", data, shuffled_results)
        sanity_report = {"phase": 6, "split": "test", "seed": 42, "integrity": integrity,
                         "primary_saved_checksum": _sha(output / "primary_saved.json"), "comparisons": comparisons,
                         "fixed_models_no_fitting_or_retuning": True, "delta_definition": "shuffled minus original",
                         "interpretation_boundary": "Single prespecified context permutation; interpret with answer-only control; no required decline or model selection"}
        write_json(output / "shuffled_metrics.json", sanity_report)
        # Verify evaluation never mutated any locked training/model/selection artifacts.
        verify_lock(root)
        execution.update(status="complete", completed_utc=_now(), primary_outputs_saved_before_shuffle=True,
                         test_evaluation_count=1, shuffled_evaluation_count=1)
        write_json(attempt, execution)
        return {"primary": primary, "shuffled": sanity_report}
    except Exception as exc:
        execution.update(status="failed", failure_utc=_now(), error=f"{type(exc).__name__}: {exc}")
        write_json(attempt, execution)
        raise
