"""Synthetic, network-free tests for validation-only Phase 4 selection."""
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from hallucination_detector.baselines.control import answer_surface_features, CONTROL_SCHEMA
from hallucination_detector.baselines.selection import (
    select_cosine, select_threshold, select_logistic, validation_metrics,
)
from hallucination_detector.baselines.pipeline import (
    load_development_split, load_models, run_phase4, save_models, schema_fingerprint,
)
from hallucination_detector.config import load_config
from hallucination_detector.features.schema import load_feature_schema
from hallucination_detector.features.preprocessing import PREPROCESSING_VERSION
from hallucination_detector.features.scaler import fit_scaler, transform_with_scaler


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        yield Path(directory)


def test_baseline_tolerance_is_numeric():
    tolerance = load_config()["baselines"]["tol"]
    assert isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool)
    assert tolerance == 1e-5


@pytest.mark.parametrize("direction,similarity", [
    ("higher", [0.1, 0.2, 0.8, 0.9]), ("lower", [0.9, 0.8, 0.2, 0.1])])
def test_cosine_direction_and_continuous_auc(direction, similarity):
    selected = select_cosine(np.array([0, 0, 1, 1]), np.array(similarity), ["lower", "higher"])["selected"]
    assert selected["direction"] == direction
    assert selected["validation_metrics"]["f1"] == 1
    assert selected["validation_metrics"]["roc_auc"] == 1
    sign = selected["score_multiplier"]
    np.testing.assert_array_equal(sign * np.array(similarity) >= selected["score_threshold"], [0, 0, 1, 1])


def test_cosine_direction_tie_prefers_original_direction():
    result = select_cosine(np.array([0, 1]), np.ones(2), ["higher", "lower"])
    assert result["selected"]["direction"] == "lower"


def test_threshold_includes_boundary_and_ties_are_deterministic():
    labels, scores = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    # Both .5 and .8 classify perfectly: reference proximity chooses .5.
    assert select_threshold(labels, scores) == 0.5
    assert select_threshold(labels, scores, reference=1) == 0.8
    assert validation_metrics(np.array([0, 1]), np.array([0.4, 0.5]), 0.5)["f1"] == 1
    # With constant scores, all positive is best; ties never separate identical scores.
    assert select_threshold(np.array([0, 1]), np.array([0.4, 0.4])) == 0.4


def test_threshold_matches_bruteforce_candidate_search():
    rng = np.random.default_rng(17)
    labels = np.tile([0, 1], 20)
    scores = np.round(rng.random(40), 1)
    candidates = np.unique(np.r_[scores, 0.5, np.nextafter(scores.max(), np.inf)])
    best = min(candidates, key=lambda t: (-validation_metrics(labels, scores, t)["f1"], abs(t-.5), t))
    assert select_threshold(labels, scores) == best


def test_positive_class_and_auc_use_scores_not_predictions():
    metrics = validation_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.2, 0.4]), 0.25)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == .5
    assert metrics["roc_auc"] == .75


class FakeLogistic:
    """Expose fitting inputs and supply prescribed validation scores for C selection."""
    fitted = []
    score_sets = {}

    def __init__(self, C, **kwargs):
        self.C = C
        self.classes_ = np.array([0, 1])
        self.n_iter_ = np.array([3])

    def fit(self, x, y):
        self.fitted.append((x.copy(), y.copy()))
        return self

    def predict_proba(self, x):
        p = np.array(self.score_sets[self.C])
        return np.column_stack([1-p, p])


def test_c_grid_threshold_selection_and_train_only_fit(monkeypatch):
    from hallucination_detector.baselines import selection
    FakeLogistic.fitted = []
    FakeLogistic.score_sets = {0.01: [.4, .3, .2, .1], .1: [.05, .1, .2, .3],
                              1.: [.05, .1, .2, .3], 10.: [.1, .2, .4, .3]}
    monkeypatch.setattr(selection, "LogisticRegression", FakeLogistic)
    train_x, val_x = np.zeros((6, 1543)), np.ones((4, 1543)) * 99
    train_y, val_y = np.tile([0, 1], 3), np.array([0, 0, 1, 1])
    model, result = select_logistic(train_x, train_y, val_x, val_y, load_config()["baselines"])
    assert [r["C"] for r in result["search"]] == [.01, .1, 1., 10.]
    assert result["selected"]["C"] == .1  # F1/AUC tie broken by smaller C
    assert result["selected"]["threshold"] == .2  # validation selected, not .5
    assert result["selected"]["validation_metrics"]["f1"] == 1
    assert all(np.array_equal(x, train_x) and np.array_equal(y, train_y) for x, y in FakeLogistic.fitted)
    assert len(FakeLogistic.fitted) == 4 and model.C == .1


def test_logistic_f1_tie_uses_auc_before_c(monkeypatch):
    from hallucination_detector.baselines import selection
    FakeLogistic.fitted = []
    FakeLogistic.score_sets = {.01: [.1, .2, .6, .3, .4, .5],
                              .1: [.1, .2, .3, .25, .4, .5]}
    monkeypatch.setattr(selection, "LogisticRegression", FakeLogistic)
    cfg = {**load_config()["baselines"], "c_grid": [.01, .1]}
    labels = np.array([0, 0, 0, 1, 1, 1])
    _, result = select_logistic(np.ones((6, 2)), labels, np.zeros((6, 2)), labels, cfg)
    assert result["search"][0]["validation_metrics"]["f1"] == result["search"][1]["validation_metrics"]["f1"]
    assert result["search"][1]["validation_metrics"]["roc_auc"] > result["search"][0]["validation_metrics"]["roc_auc"]
    assert result["selected"]["C"] == .1


def test_answer_only_feature_contract():
    np.testing.assert_array_equal(answer_surface_features(["Hi, 42!", ""]),
                                  [[7, 2, 1, 1, 1, 2], [0, 0, 0, 0, 0, 0]])
    row = {"candidate_answer": "Same answer.", "question": object(), "context": object(), "source_id": object()}
    features = answer_surface_features([row["candidate_answer"]])
    row.update(question="unrelated", context="changed", source_id="changed")
    np.testing.assert_array_equal(features, answer_surface_features([row["candidate_answer"]]))
    with pytest.raises(TypeError):
        answer_surface_features([row])


def test_refuse_test_split_before_access():
    with pytest.raises(ValueError, match="forbidden"):
        load_development_split(Path("does-not-exist"), "test", 1543)


def _write_fixture(root):
    schema = load_feature_schema()
    def write(name, data):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data), encoding="utf-8")
    write("configs/feature_schema.json", schema)
    write("data/features/schema.json", schema)
    write("data/metadata/halueval_qa_provenance.json", {"sha256": "data-sha", "source_revision": "source-rev"})
    write("data/splits/halueval_original_split_manifest.json", [])
    from hallucination_detector.baselines.pipeline import _sha
    write("data/embedding_cache/cache_metadata_questions.json", {
        "preprocessing_version": PREPROCESSING_VERSION, "feature_schema_version": schema["schema_version"],
        "dataset_checksum": "data-sha", "split_manifest_checksum": _sha(root / "data/splits/halueval_original_split_manifest.json")})
    train = np.zeros((20, 1543), dtype=np.float32)
    train[:, 0] = np.tile([-1., 1.], 10)
    train[:, 1536] = np.tile([.1, .9], 10)
    mean, scale = fit_scaler(train)
    (root / "data/scaler").mkdir()
    np.savez(root / "data/scaler/scaler.npz", mean=mean, scale=scale)
    answers = {}
    for split, n in (("train", 20), ("validation", 8)):
        labels = np.tile([0, 1], n//2)
        x = train[:n].copy()
        rows = [dict(source_id=f"{split}-{i//2}", example_id=f"{split}-{i}", split=split, label=int(label),
                     candidate_answer=("yes" if label == 0 else "a long answer with 42 words!"),
                     question="forbidden predictor", reference_context="forbidden predictor") for i, label in enumerate(labels)]
        if split == "validation":
            rows[0]["candidate_answer"] = "very " * 100
        p = root / "data/features" / split
        p.mkdir(exist_ok=True)
        for name, a in (("features", x), ("features_scaled", transform_with_scaler(x, mean, scale)), ("labels", labels)):
            np.save(p / f"{name}.npy", a)
        for name, field in (("example_ids", "example_id"), ("source_ids", "source_id")):
            write(f"data/features/{split}/{name}.json", [r[field] for r in rows])
        proc = root / "data/processed" / f"halueval_expanded_{split}.jsonl"
        proc.parent.mkdir(exist_ok=True)
        proc.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        answers[split] = [r["candidate_answer"] for r in rows]
    return schema, answers


def test_full_workflow_never_opens_test_and_roundtrips_models(workspace, monkeypatch):
    schema, answers = _write_fixture(workspace)
    original_load, original_read = np.load, Path.read_text
    def guard(path):
        assert "test" not in Path(path).parts
        assert "halueval_expanded_test.jsonl" not in str(path)
    def guarded_load(path, *args, **kwargs):
        guard(path)
        return original_load(path, *args, **kwargs)
    def guarded_read(path, *args, **kwargs):
        guard(path)
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded_load)
    monkeypatch.setattr(Path, "read_text", guarded_read)
    report = run_phase4(workspace)
    assert report["row_counts"] == {"train": 20, "validation": 8}
    assert report["evaluation_split"] == "validation"
    assert report["primary_feature_dim"] == 1543
    bundle = load_models(workspace / "checkpoints/phase4/baselines.joblib", schema)
    assert bundle["primary_model"].n_features_in_ == 1543
    assert bundle["control_model"].n_features_in_ == 6
    scaler = bundle["control_scaler"]
    np.testing.assert_array_equal(scaler.mean_, answer_surface_features(answers["train"]).mean(axis=0))
    assert scaler.n_samples_seen_ == 20
    assert not np.array_equal(scaler.mean_, answer_surface_features(answers["validation"]).mean(axis=0))
    p = workspace / "copy.joblib"
    save_models(p, bundle)
    restored = load_models(p, schema)
    x = original_load(workspace / "data/features/validation/features_scaled.npy")
    np.testing.assert_array_equal(bundle["primary_model"].predict_proba(x), restored["primary_model"].predict_proba(x))
    wrong = copy.deepcopy(schema)
    wrong["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="schema"):
        load_models(p, wrong)


def test_reject_mismatched_input_schema(workspace):
    schema, _ = _write_fixture(workspace)
    schema["total_dim"] = 1537
    (workspace / "data/features/schema.json").write_text(json.dumps(schema))
    with pytest.raises(ValueError, match="schema"):
        run_phase4(workspace)
