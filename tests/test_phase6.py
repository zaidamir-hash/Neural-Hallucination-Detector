"""Offline Phase 6 compatibility, fixed inference, persistence and context sanity tests."""
import copy
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

from hallucination_detector.baselines.pipeline import _json, _sha, run_phase4
from hallucination_detector.baselines.selection import validation_metrics
from hallucination_detector.config import load_config
from hallucination_detector.context_sanity import (
    apply_context_mapping, context_mapping, load_frozen_cache, recompute_context_features,
)
from hallucination_detector.data.split_phase2 import _context_group_id, deterministic_example_id
from hallucination_detector.data.validate import compute_source_id
from hallucination_detector.evaluation import (
    LOCK_PATH, METRICS, aggregate_mlp, load_evaluation_split, load_fixed_models,
    persist_predictions, predict_mlp, prepare_evaluation, score_models,
    validate_evaluation_alignment, verify_lock,
)
from hallucination_detector.evaluation_pipeline import run_final_evaluation
from hallucination_detector.features.primary import build_primary_features
from hallucination_detector.features.scaler import load_scaler, transform_with_scaler
from hallucination_detector.mlp import HallucinationMLP
from hallucination_detector.mlp_pipeline import load_phase5_inputs, write_json
from tests.test_phase4 import _write_fixture


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def source_rows(n=6):
    rows, manifest = [], []
    for i in range(n):
        # Two sources share normalized context, including case/whitespace differences.
        context = f"place {i} number {i}"
        if i == 1:
            context = " PLACE   0 number 0 "
        original = {"question": f"question {i}", "knowledge": context,
                    "right_answer": f"place {i}", "hallucinated_answer": f"number {i+100}"}
        sid = compute_source_id(original)
        manifest.append({"source_id": sid, "split": "test", "duplicate_context_group_id": _context_group_id(context)})
        for label, variant, name in ((0, "right_answer", "Grounded"), (1, "hallucinated_answer", "Hallucinated")):
            rows.append({"source_id": sid, "example_id": deterministic_example_id(sid, variant), "split": "test",
                         "label": label, "label_name": name, "answer_variant": variant,
                         "question": original["question"], "candidate_answer": original[variant], "reference_context": context})
    return rows, manifest


def synthetic_cache(rows):
    from hallucination_detector.features.preprocessing import normalize_text
    rng = np.random.default_rng(7)
    def vector():
        x = rng.normal(size=384).astype(np.float32)
        return x / np.linalg.norm(x)
    cache = {}
    for key, field in (("questions", "question"), ("answers", "candidate_answer"), ("chunks", "reference_context")):
        cache[key] = {text: (np.stack([vector(), vector(), vector(), vector()]) if key == "chunks" else vector())
                      for text in sorted({normalize_text(r[field]) for r in rows})}
    return cache


@pytest.fixture
def phase6_workspace(tmp_path):
    _write_fixture(tmp_path)
    rows, manifest = source_rows()
    write_json(tmp_path / "data/splits/halueval_original_split_manifest.json", manifest)
    meta_path = tmp_path / "data/embedding_cache/cache_metadata_questions.json"
    meta = {**_json(meta_path), "split_manifest_checksum": _sha(tmp_path / "data/splits/halueval_original_split_manifest.json"),
            "cache_format_version": "1.0.0", "encoder_name": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dim": 384,
            "normalized_embeddings": True, "context_chunk_max_tokens": 192, "context_chunk_overlap_tokens": 32}
    for variant in ("questions", "answers", "contexts", "chunks"):
        write_json(tmp_path / f"data/embedding_cache/cache_metadata_{variant}.json", meta)
    run_phase4(tmp_path)
    splits, compatibility = load_phase5_inputs(tmp_path)
    settings = load_config()["mlp"]
    runs = []
    for seed in (101, 202, 303):
        torch.manual_seed(seed)
        model = HallucinationMLP(settings)
        path = f"checkpoints/phase5/mlp_seed_{seed}.pt"
        (tmp_path / path).parent.mkdir(exist_ok=True)
        threshold = .4 + seed/10000
        bundle = {"state_dict": model.state_dict(), "configuration": settings, "compatibility": compatibility,
                  "seed": seed, "threshold": threshold, "best_epoch": 2, "best_validation_loss": .7}
        torch.save(bundle, tmp_path / path)
        metrics = validation_metrics(splits["validation"]["labels"], predict_mlp(model, splits["validation"]["features_scaled"]), threshold)
        runs.append({k: bundle[k] for k in ("seed", "threshold", "best_epoch", "best_validation_loss")})
        runs[-1].update(checkpoint_path=path, checkpoint_checksum=_sha(tmp_path / path), validation_metrics=metrics)
    write_json(tmp_path / "results/metrics/phase5_validation.json", {
        "seeds": [101, 202, 303], "configuration": settings, "evaluation_split": "validation",
        "provenance": compatibility, "runs": runs, "summary": {"representative_seed": 202}})
    root = Path(__file__).resolve().parents[1]
    for directory in ("src/hallucination_detector", "tests"):
        for src in (root / directory).rglob("*.py"):
            dst = tmp_path / src.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    for relative in ("configs/default.yaml", "scripts/run_phase6.py"):
        dst = tmp_path / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, dst)
    # Test data remain unopened by preparation, and contain no downloaded data.
    cache = synthetic_cache(rows)
    for variant, mapping in cache.items():
        write_json(tmp_path / f"data/embedding_cache/embedding_ids_{variant}.json", list(mapping))
        arrays = list(mapping.values())
        embeddings = np.concatenate(arrays) if variant == "chunks" else np.stack(arrays)
        np.save(tmp_path / f"data/embedding_cache/embeddings_{variant}.npy", embeddings)
        if variant == "chunks":
            np.save(tmp_path / "data/embedding_cache/chunk_offsets.npy", np.arange(len(arrays)+1, dtype=np.int64)*4)
    raw = recompute_context_features(rows, cache)
    mean, scale = load_scaler(tmp_path / "data/scaler/scaler.npz")
    directory = tmp_path / "data/features/test"
    directory.mkdir()
    for name, value in (("features", raw), ("features_scaled", transform_with_scaler(raw, mean, scale)),
                        ("labels", np.array([r["label"] for r in rows]))):
        np.save(directory / f"{name}.npy", value)
    for key in ("example_id", "source_id"):
        write_json(directory / f"{key}s.json", [r[key] for r in rows])
    write_json(directory / "metadata.json", {"split": "test", "feature_dim": 1543, "row_count": len(rows),
                                            "label_counts": {"grounded": len(rows)//2, "hallucinated": len(rows)//2}})
    (tmp_path / "data/processed/halueval_expanded_test.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return tmp_path


def test_lock_and_development_smoke_before_any_test_input(phase6_workspace, monkeypatch):
    root = phase6_workspace
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert "test" not in path.parts
        assert path.name != "halueval_expanded_test.jsonl"
        assert not path.name.startswith(("embedding_ids_", "embeddings_"))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(LogisticRegression, "fit", lambda *a, **k: pytest.fail("No fitting authorized"))
    monkeypatch.setattr(StandardScaler, "fit", lambda *a, **k: pytest.fail("No scaler fitting authorized"))
    smoke = prepare_evaluation(root)
    assert smoke["all_saved_validation_metrics_reproduced"] and not smoke["test_inputs_opened"]
    lock = verify_lock(root)
    assert lock["decisions"]["representative_mlp_seed"] == 202
    assert lock["decisions"]["shuffled_context_seed"] == 42
    assert lock["metric_definitions"]["mlp_sd_ddof"] == 1
    assert lock["compatibility"]["preprocessing_version"] == "2.0.0"
    with pytest.raises(FileExistsError):
        prepare_evaluation(root)


@pytest.mark.parametrize("target", ["data/scaler/scaler.npz", "configs/feature_schema.json",
                                  "results/metrics/phase4_validation.json", "checkpoints/phase5/mlp_seed_202.pt"])
def test_changed_locked_input_rejected_before_test(phase6_workspace, target):
    root = phase6_workspace
    prepare_evaluation(root)
    path = root / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        load_evaluation_split(root, _json(root / LOCK_PATH))


@pytest.mark.parametrize("kind", ["threshold", "schema", "scaler", "checkpoint"])
def test_checkpoint_report_incompatibility(phase6_workspace, kind):
    import joblib
    root = phase6_workspace
    phase4 = _json(root / "results/metrics/phase4_validation.json")
    phase5 = _json(root / "results/metrics/phase5_validation.json")
    if kind == "checkpoint":
        phase5["runs"][0]["checkpoint_checksum"] = "changed"
    else:
        path = root / "checkpoints/phase4/baselines.joblib"
        bundle = joblib.load(path)
        if kind == "threshold":
            bundle["primary_threshold"] += .1
        elif kind == "schema":
            bundle["schema_fingerprint"] = "changed"
        else:
            bundle["primary_scaler"]["mean"][0] += 1
        joblib.dump(bundle, path)
    with pytest.raises(ValueError):
        load_fixed_models(root, phase4, phase5)


def test_example_label_alignment_and_variant_integrity():
    rows, manifest = source_rows()
    ids, sources, labels = [r["example_id"] for r in rows], [r["source_id"] for r in rows], np.array([r["label"] for r in rows])
    validate_evaluation_alignment(rows, ids, sources, labels, manifest)
    for modified_ids, modified_labels in ((ids[::-1], labels), (ids, 1-labels)):
        with pytest.raises(ValueError, match="alignment"):
            validate_evaluation_alignment(rows, modified_ids, sources, modified_labels, manifest)
    changed = copy.deepcopy(rows)
    changed[0]["answer_variant"] = "hallucinated_answer"
    with pytest.raises(ValueError, match="variant"):
        validate_evaluation_alignment(changed, ids, sources, labels, manifest)
    changed = copy.deepcopy(rows)
    changed[0]["candidate_answer"] = "different text"
    with pytest.raises(ValueError, match="identity"):
        validate_evaluation_alignment(changed, ids, sources, labels, manifest)


def test_metric_direction_threshold_and_persisted_continuous_scores(tmp_path):
    labels, scores = np.array([0, 0, 1, 1]), np.array([.1, .3, .2, .4])
    metrics = validation_metrics(labels, scores, .25)
    assert metrics == {"accuracy": .5, "precision": .5, "recall": .5, "f1": .5,
                       "roc_auc": .75, "confusion_matrix": [[1, 1], [1, 1]]}
    data = {"example_ids": list("abcd"), "source_ids": ["a", "a", "b", "b"], "labels": labels}
    path = tmp_path / "predictions.csv"
    persist_predictions(path, data, [{"model": "mlp", "seed": 202, "threshold": .3, "scores": scores}])
    records = list(csv.DictReader(path.open()))
    assert [r["example_id"] for r in records] == list("abcd")
    assert [float(r["continuous_score"]) for r in records] == scores.tolist()
    assert [int(r["predicted_label"]) for r in records] == [0, 1, 0, 1]  # inclusive boundary
    assert all(r["model"] == "mlp" and r["seed"] == "202" for r in records)


def test_three_seed_summary_sample_sd():
    results = [{"model": "mlp", "seed": seed, "metrics": {k: value for k in (*METRICS, "confusion_matrix")}}
               for seed, value in ((101, .7), (202, .8), (303, .9))]
    summary = aggregate_mlp(results)
    assert summary["ddof"] == 1
    assert summary["metrics"]["f1"]["mean"] == pytest.approx(.8)
    assert summary["metrics"]["f1"]["sample_sd"] == pytest.approx(.1)
    with pytest.raises(ValueError):
        aggregate_mlp(results[:2])


@pytest.mark.parametrize("direction,sign", [("higher", 1), ("lower", -1)])
def test_fixed_cosine_direction_and_threshold(direction, sign):
    class FixedLR:
        def predict_proba(self, x):
            return np.column_stack([np.ones(len(x))*.4, np.ones(len(x))*.6])
    x = np.zeros((4, 1543), dtype=np.float32)
    x[:, 1536] = sign * np.array([.1, .3, .2, .4])
    data = {"features": x, "features_scaled": x, "labels": np.array([0, 0, 1, 1])}
    models = {"baseline": {"primary_model": FixedLR()}, "mlps": {}}
    decisions = {"cosine": {"direction": direction, "score_multiplier": sign, "score_threshold": .25},
                 "primary_logistic": {"threshold": .7}, "mlp_runs": []}
    results = score_models(data, models, decisions, sanity=True)
    assert results[0]["threshold"] == .25
    assert results[0]["metrics"]["roc_auc"] == .75
    assert results[0]["metrics"]["confusion_matrix"] == [[1, 1], [1, 1]]
    assert results[1]["metrics"]["recall"] == 0  # saved threshold, never model.predict or .5


@pytest.mark.parametrize("sizes", [(3, 3), (2, 2, 2), (3, 2, 1), (1, 1), (1, 1, 1, 1, 1)])
def test_shuffle_handles_duplicate_groups_at_feasibility_boundary(sizes):
    rows = []
    for group, size in enumerate(sizes):
        for index in range(size):
            rows.extend({"source_id": f"{group}-{index}", "label": label, "reference_context": f"context {group}"}
                        for label in (0, 1))
    for seed in range(10):
        _, integrity = context_mapping(rows, seed)
        assert integrity["all_passed"]


def test_shuffle_reproducible_bijective_duplicate_exclusion_and_paired_inputs():
    rows, _ = source_rows()
    mapping, integrity = context_mapping(rows, 42)
    assert (mapping, integrity) == context_mapping(rows[::-1], 42)
    assert all(integrity["checks"].values())
    assert integrity["largest_context_group"] == 2
    assert len(mapping) == len(set(mapping.values())) == 6
    shuffled = apply_context_mapping(rows, mapping)
    for original, changed in zip(rows, shuffled):
        assert {k: v for k, v in original.items() if k != "reference_context"} == {k: v for k, v in changed.items() if k != "reference_context"}
    for i in range(0, len(rows), 2):
        assert shuffled[i]["reference_context"] == shuffled[i+1]["reference_context"]
    with pytest.raises(ValueError, match="identical"):
        apply_context_mapping(rows, {s: s for s in mapping})
    with pytest.raises(ValueError, match="No valid"):
        context_mapping(rows[:4], 42)
    with pytest.raises(ValueError, match="pairs"):
        context_mapping(rows[:-1], 42)


def test_every_context_feature_recomputed_and_qa_unchanged():
    rows, _ = source_rows()
    cache = synthetic_cache(rows)
    original = recompute_context_features(rows, cache)
    mapping, _ = context_mapping(rows)
    shuffled_rows = apply_context_mapping(rows, mapping)
    shuffled = recompute_context_features(shuffled_rows, cache)
    from hallucination_detector.features.preprocessing import normalize_text
    for index, row in enumerate(shuffled_rows):
        q, a, c = (normalize_text(row[k]) for k in ("question", "candidate_answer", "reference_context"))
        q_emb, a_emb, chunks = cache["questions"][q], cache["answers"][a], cache["chunks"][c]
        best = chunks[np.argmax(chunks @ a_emb)]
        expected = build_primary_features(q_emb, a_emb, best, chunks, a, c, np.max(np.einsum("ij,j->i", chunks, q_emb)))
        np.testing.assert_array_equal(shuffled[index], expected)
    qa = np.r_[np.arange(768, 1536), 1538, 1542]
    np.testing.assert_array_equal(original[:, qa], shuffled[:, qa])
    for indices in (slice(0, 768), 1536, 1537, 1539, 1540):
        assert not np.array_equal(original[:, indices], shuffled[:, indices])
    # Explicit lexical/numeric coverage and answer-dependent chunk reselection fixture.
    a, q = np.eye(384, dtype=np.float32)[:2]
    b = np.eye(384, dtype=np.float32)[2]
    one = [{"question": "q", "candidate_answer": "number 7", "reference_context": "number 7"}]
    two = [{**one[0], "reference_context": "other 9"}]
    cache = {"questions": {"q": q}, "answers": {"number 7": a},
             "chunks": {"number 7": np.stack([q, a]), "other 9": np.stack([b, -a])}}
    before, after = recompute_context_features(one, cache)[0], recompute_context_features(two, cache)[0]
    np.testing.assert_array_equal(before[:384], abs(a-a))
    np.testing.assert_array_equal(after[:384], abs(a-b))
    assert before[1536] == 1 and after[1536] == 0
    assert before[1540] == before[1541] == 1 and after[1540] == after[1541] == 0


def test_cache_rejects_offsets_and_normalization(phase6_workspace):
    root = phase6_workspace
    meta = _json(root / "results/metrics/phase5_validation.json")["provenance"]["encoder_metadata"]
    assert load_frozen_cache(root, meta)
    path = root / "data/embedding_cache/chunk_offsets.npy"
    offsets = np.load(path)
    offsets[1] = 0
    np.save(path, offsets)
    with pytest.raises(ValueError, match="offsets"):
        load_frozen_cache(root, meta)


def test_final_loader_requires_smoke_and_unchanged_training_scaler(phase6_workspace):
    root = phase6_workspace
    with pytest.raises(FileNotFoundError):
        load_evaluation_split(root, {})
    prepare_evaluation(root)
    lock = verify_lock(root)
    data = load_evaluation_split(root, lock)
    mean, scale = load_scaler(root / "data/scaler/scaler.npz")
    np.testing.assert_array_equal(data["features_scaled"], transform_with_scaler(data["features"], mean, scale))
    path = root / "data/features/test/features_scaled.npy"
    np.save(path, data["features_scaled"] + 1)
    with pytest.raises(ValueError, match="training scaler"):
        load_evaluation_split(root, lock)


def test_full_final_and_shuffled_workflow_never_fits_or_selects(phase6_workspace, monkeypatch):
    from hallucination_detector import evaluation_pipeline
    from hallucination_detector.baselines import selection
    from hallucination_detector.features import scaler
    root = phase6_workspace
    prepare_evaluation(root)
    def forbidden(*args, **kwargs):
        pytest.fail("Final evaluation must never fit, train or tune")
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(scaler, "fit_scaler", forbidden)
    monkeypatch.setattr(selection, "select_threshold", forbidden)
    monkeypatch.setattr(selection, "select_cosine", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    class GitResult:
        stdout = "synthetic-commit\n"
    monkeypatch.setattr(evaluation_pipeline.subprocess, "run", lambda *a, **k: GitResult())
    original_mapping = evaluation_pipeline.context_mapping
    def after_primary_saved(*args, **kwargs):
        for name in ("primary_metrics.json", "primary_predictions.csv", "comparison.md", "primary_saved.json"):
            assert (root / "results/metrics/phase6" / name).is_file()
        return original_mapping(*args, **kwargs)
    monkeypatch.setattr(evaluation_pipeline, "context_mapping", after_primary_saved)
    report = run_final_evaluation(root)
    primary, sanity = report["primary"], report["shuffled"]
    assert primary["representative_mlp_seed"] == 202
    assert len(primary["results"]) == 6
    assert [(r["model"], r["seed"]) for r in sanity["comparisons"]] == [("cosine", None), ("primary_logistic", None), ("mlp", 202)]
    assert all(sanity["integrity"]["checks"].values())
    for result in sanity["comparisons"]:
        assert result["delta_shuffled_minus_original"]["f1"] == result["shuffled"]["f1"] - result["original"]["f1"]
    with (root / "results/metrics/phase6/primary_predictions.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    ids = [r["example_id"] for r in predictions[:12]]
    for start in range(0, len(predictions), 12):
        assert [r["example_id"] for r in predictions[start:start+12]] == ids
    for figure in primary["figures"]:
        assert (root / figure).stat().st_size > 1000
    assert verify_lock(root)
    assert _json(root / "results/metrics/phase6/execution.json")["reruns"] == []
    with pytest.raises(FileExistsError, match="already attempted"):
        run_final_evaluation(root)
