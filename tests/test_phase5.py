"""Small offline fixtures for Phase 5 determinism and locked validation policy."""
import copy
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

from hallucination_detector.config import load_config
from hallucination_detector.baselines.pipeline import run_phase4, _sha
from hallucination_detector.baselines.selection import select_threshold, validation_metrics
from hallucination_detector.mlp import (
    HallucinationMLP, seed_everything, epoch_loss, make_loader, train_seed,
    predict_validation, load_checkpoint, summarize_runs,
)
from hallucination_detector.mlp_pipeline import load_phase5_inputs, run_phase5, validate_training_config
from tests.test_phase4 import _write_fixture


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def small_data():
    rng = np.random.default_rng(19)
    result = []
    for split, n in (("train", 10), ("validation", 6)):
        x = rng.normal(size=(n, 1543)).astype(np.float32)
        y = np.tile([0, 1], n // 2)
        result.append({"split": split, "features_scaled": x, "labels": y})
    return result


def small_settings():
    return {**load_config()["mlp"], "batch_size": 4, "max_epochs": 3, "early_stopping_patience": 2}


def test_locked_numeric_config_and_architecture():
    config = load_config()
    validate_training_config(config)
    assert isinstance(config["mlp"]["learning_rate"], float)
    assert isinstance(config["mlp"]["weight_decay"], float)
    model = HallucinationMLP(config["mlp"])
    assert [(m.in_features, m.out_features) for m in model.modules() if isinstance(m, nn.Linear)] == [(1543, 256), (256, 64), (64, 1)]
    assert [m.p for m in model.modules() if isinstance(m, nn.Dropout)] == [.2, .1]
    assert sum(isinstance(m, nn.ReLU) for m in model.modules()) == 2
    with torch.no_grad():
        model.layers[-1].weight.zero_()
        model.layers[-1].bias.fill_(-3)
    assert model(torch.zeros(1, 1543)).shape == (1,)
    assert torch.equal(model(torch.zeros(5, 1543)), torch.full((5,), -3.))  # raw logits
    changed = copy.deepcopy(config)
    changed["mlp"]["learning_rate"] = "0.001"
    with pytest.raises(ValueError, match="configuration"):
        validate_training_config(changed)


def test_seeded_initialization_and_all_rngs():
    def draw(seed):
        seed_everything(seed)
        return random.random(), np.random.random(), HallucinationMLP(load_config()["mlp"]).state_dict()
    a, b = draw(101), draw(101)
    assert a[:2] == b[:2]
    assert all(torch.equal(a[2][k], b[2][k]) for k in a[2])
    c = draw(202)
    assert any(not torch.equal(a[2][k], c[2][k]) for k in a[2])


def test_sample_weighted_loss_incomplete_batch_and_eval():
    class Echo(nn.Module):
        def forward(self, x):
            assert not self.training
            assert not torch.is_grad_enabled()
            return x[:, 0]
    x = torch.tensor([[0.], [0.], [0.], [0.], [10.]])
    y = torch.zeros(5)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    actual = epoch_loss(Echo(), loader, "cpu")
    expected = nn.BCEWithLogitsLoss()(x[:, 0], y).item()
    assert actual == pytest.approx(expected, abs=1e-6)
    wrong = (nn.BCEWithLogitsLoss()(x[:4, 0], y[:4]).item() + nn.BCEWithLogitsLoss()(x[4:, 0], y[4:]).item()) / 2
    assert abs(actual - wrong) > 1


def test_shuffling_only_training_and_seeded(small_data):
    train, validation = small_data
    loaders = [make_loader(train["features_scaled"], train["labels"], "train", 4, 101) for _ in range(2)]
    assert isinstance(loaders[0].sampler, RandomSampler)
    assert list(loaders[0].sampler) == list(loaders[1].sampler)
    loader = make_loader(validation["features_scaled"], validation["labels"], "validation", 4, 101)
    assert isinstance(loader.sampler, SequentialSampler)
    assert list(loader.sampler) == list(range(6))


def test_deterministic_cpu_training_reload_and_validation_threshold(tmp_path, small_data):
    settings = small_settings()
    runs = [train_seed(*small_data, 101, settings, tmp_path / f"{i}.pt", {"schema": "fixture"}) for i in range(2)]
    assert runs[0] == runs[1]
    model, bundle = load_checkpoint(tmp_path / "0.pt", {"schema": "fixture"})
    other, _ = load_checkpoint(tmp_path / "1.pt", {"schema": "fixture"})
    scores = predict_validation(model, small_data[1]["features_scaled"], batch_size=4)
    loader = make_loader(small_data[1]["features_scaled"], small_data[1]["labels"], "validation", settings["batch_size"], 101)
    assert epoch_loss(model, loader, "cpu") == runs[0]["best_validation_loss"]
    np.testing.assert_array_equal(scores, predict_validation(other, small_data[1]["features_scaled"], batch_size=4))
    threshold = select_threshold(small_data[1]["labels"], scores)
    assert threshold == runs[0]["threshold"] == bundle["threshold"]
    assert runs[0]["validation_metrics"] == validation_metrics(small_data[1]["labels"], scores, threshold)
    with pytest.raises(ValueError, match="Incompatible"):
        load_checkpoint(tmp_path / "0.pt", {"schema": "wrong"})


def test_early_stopping_restores_best_and_keeps_earliest_tie(tmp_path, small_data, monkeypatch):
    from hallucination_detector import mlp
    validation_losses = iter([.7, .4, .4, .8])
    saved_states = []
    def fake_epoch(model, loader, device, optimizer=None):
        if optimizer is not None:
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(.01)
            saved_states.append(copy.deepcopy(model.state_dict()))
            return .5
        return next(validation_losses)
    monkeypatch.setattr(mlp, "epoch_loss", fake_epoch)
    settings = {**small_settings(), "max_epochs": 10}
    path = tmp_path / "best.pt"
    run = train_seed(*small_data, 101, settings, path, {})
    restored, bundle = load_checkpoint(path, {})
    assert run["epochs_completed"] == 4
    assert run["best_epoch"] == bundle["best_epoch"] == 2
    assert run["best_validation_loss"] == .4
    assert run["stopped_early"]
    assert [h["improved"] for h in run["history"]] == [True, True, False, False]
    assert all(torch.equal(restored.state_dict()[k], saved_states[1][k]) for k in saved_states[1])
    assert any(not torch.equal(restored.state_dict()[k], saved_states[-1][k]) for k in saved_states[-1])


def test_representative_seed_median_ties_and_sample_std():
    metrics = validation_metrics(np.array([0, 1]), np.array([.1, .9]), .5)
    def run(seed, f1):
        return {"seed": seed, "validation_metrics": {**metrics, "f1": f1}}
    result = summarize_runs([run(101, .9), run(202, .7), run(303, .8)])
    assert result["representative_seed"] == 303
    assert result["validation_metrics"]["f1"]["mean"] == pytest.approx(.8)
    assert result["validation_metrics"]["f1"]["std"] == pytest.approx(.1)
    assert summarize_runs([run(303, .9), run(202, .9), run(101, .8)])["representative_seed"] == 202
    assert summarize_runs([run(303, .9), run(202, .9), run(101, .9)])["representative_seed"] == 101


@pytest.mark.parametrize("split", ["test", "TEST", "holdout"])
def test_reject_test_roles_before_array_access(split, small_data, tmp_path):
    with pytest.raises(ValueError, match="forbidden"):
        make_loader(None, None, split, 4, 101)
    with pytest.raises(ValueError, match="forbidden"):
        predict_validation(None, None, split=split)
    with pytest.raises(ValueError, match="forbidden"):
        train_seed({"split": split}, small_data[1], 101, small_settings(), tmp_path / "never.pt", {})
    with pytest.raises(ValueError, match="forbidden"):
        train_seed(small_data[0], {"split": split}, 101, small_settings(), tmp_path / "never.pt", {})
    assert not (tmp_path / "never.pt").exists()


@pytest.fixture
def phase5_workspace(tmp_path):
    _write_fixture(tmp_path)
    report = run_phase4(tmp_path)
    cache = {**report["provenance"]["encoder"], "cache_format_version": "1.0.0",
             "encoder_name": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dim": 384,
             "normalized_embeddings": True, "context_chunk_max_tokens": 192,
             "context_chunk_overlap_tokens": 32}
    for variant in ("questions", "answers", "contexts", "chunks"):
        (tmp_path / f"data/embedding_cache/cache_metadata_{variant}.json").write_text(json.dumps(cache))
    report["provenance"]["encoder"] = cache
    (tmp_path / "results/metrics/phase4_validation.json").write_text(json.dumps(report))
    return tmp_path


def test_input_loading_never_fits_or_opens_test(phase5_workspace, monkeypatch):
    from hallucination_detector.features import scaler
    monkeypatch.setattr(scaler, "fit_scaler", lambda *a: pytest.fail("No scaler fit authorized"))
    original = Path.open
    def guarded_open(path, *args, **kwargs):
        assert "test" not in path.parts
        assert "halueval_expanded_test.jsonl" not in str(path)
        assert not path.name.startswith("embeddings_")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    splits, meta = load_phase5_inputs(phase5_workspace)
    assert set(splits) == {"train", "validation"}
    assert meta["primary_feature_dim"] == 1543
    assert meta["preprocessing_version"] == "2.0.0"


@pytest.mark.parametrize("target,key,value", [
    ("data/features/schema.json", "schema_version", "wrong"),
    ("data/embedding_cache/cache_metadata_answers.json", "preprocessing_version", "wrong"),
    ("data/embedding_cache/cache_metadata_chunks.json", "normalized_embeddings", False),
    ("data/metadata/halueval_qa_provenance.json", "sha256", "wrong"),
])
def test_incompatible_metadata_rejected(phase5_workspace, target, key, value):
    path = phase5_workspace / target
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incompatible"):
        load_phase5_inputs(phase5_workspace)


def test_changed_scaler_rejected(phase5_workspace):
    np.savez(phase5_workspace / "data/scaler/scaler.npz", mean=np.zeros(1543), scale=np.ones(1543))
    with pytest.raises(ValueError, match="provenance"):
        load_phase5_inputs(phase5_workspace)


def test_misaligned_labels_rejected_even_with_updated_hash(phase5_workspace):
    path = phase5_workspace / "data/features/validation/labels.npy"
    np.save(path, np.tile([1, 0], 4))
    report_path = phase5_workspace / "results/metrics/phase4_validation.json"
    report = json.loads(report_path.read_text())
    report["provenance"]["input_checksums"]["data/features/validation/labels.npy"] = _sha(path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="align"):
        load_phase5_inputs(phase5_workspace)


def test_full_three_seed_workflow(phase5_workspace, monkeypatch):
    from hallucination_detector import mlp_pipeline
    config = load_config()
    config["mlp"] = small_settings()
    # Only reduce epochs/batch size for this synthetic integration test.
    monkeypatch.setattr(mlp_pipeline, "load_config", lambda: config)
    monkeypatch.setattr(mlp_pipeline, "validate_training_config", lambda c: None)
    root = Path(__file__).resolve().parents[1]
    for relative in ("src/hallucination_detector/mlp.py", "src/hallucination_detector/mlp_pipeline.py",
                     "scripts/run_phase5.py", "configs/default.yaml"):
        destination = phase5_workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    # tmp_path may be outside the repository: supply only the read-only Git result.
    class GitResult:
        stdout = "fixture-commit\n"
    monkeypatch.setattr(mlp_pipeline.subprocess, "run", lambda *a, **k: GitResult())
    original_open = Path.open
    original_load = np.load
    def guarded_open(path, *args, **kwargs):
        assert "test" not in path.parts
        assert "halueval_expanded_test.jsonl" not in str(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    def guarded_load(path, *args, **kwargs):
        assert "test" not in Path(path).parts
        assert not Path(path).name.startswith("embeddings_")
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded_load)
    report = run_phase5(phase5_workspace)
    assert report["evaluation_split"] == "validation"
    assert [r["seed"] for r in report["runs"]] == [101, 202, 303]
    for run in report["runs"]:
        history = json.loads((phase5_workspace / run["history_path"]).read_text())
        assert len(history) == 3
        _, bundle = load_checkpoint(phase5_workspace / run["checkpoint_path"], report["provenance"])
        assert bundle["threshold"] == run["threshold"]
    assert (phase5_workspace / report["loss_figure"]).is_file()
