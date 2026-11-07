"""Offline regression checks for independent experiment workspaces."""
from pathlib import Path

import pytest

from hallucination_detector.error_analysis import attach_reviews, match_historical_reviews, select_errors
from hallucination_detector.evaluation import validation_representative
from hallucination_detector.reproduction import OUTPUT_DIRS, REFERENCE, WORKSPACE, initialize_workspace
from tests.test_phase7 import fixture_rows, notes_for


def test_initializer_isolates_outputs_and_protects_existing_target(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("README.md", "PROJECT_PLAN.md", "AGENTS.md", "pyproject.toml", "requirements.txt",
                 "constraints-tested.txt", ".gitignore", "src/hallucination_detector/example.py",
                 "data/metadata/halueval_qa_provenance.json", "results/error_analysis/phase7/analyst_notes.json",
                 "results/error_analysis/phase7/reviewed_cases.json", "data/raw/qa_data.json",
                 "results/metrics/phase6/evaluation_manifest.json", "checkpoints/model.pt"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    monkeypatch.setattr("hallucination_detector.reproduction.source_commit", lambda root: "a" * 40)
    target = tmp_path / "independent"
    initialize_workspace(source, target)
    assert (target / WORKSPACE).is_file()
    assert (target / REFERENCE / "analyst_notes.json").is_file()
    assert (target / "src/hallucination_detector/example.py").is_file()
    assert all(not list((target / directory).iterdir()) for directory in OUTPUT_DIRS)
    with pytest.raises(FileExistsError):
        initialize_workspace(source, target)
    assert (source / "checkpoints/model.pt").read_text(encoding="utf-8") == "checkpoints/model.pt"


def test_independent_representative_uses_median_validation_f1_and_tie_rule(tmp_path):
    (tmp_path / WORKSPACE).write_text("{}", encoding="utf-8")
    runs = [{"seed": seed, "validation_metrics": dict.fromkeys(
        ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"), f1)}
        for seed, f1 in ((101, .8), (202, .9), (303, .8))]
    report = {"runs": runs, "summary": {"representative_seed": 101}}
    assert validation_representative(tmp_path, report) == 101
    report["summary"]["representative_seed"] = 202
    with pytest.raises(ValueError, match="validation selection"):
        validation_representative(tmp_path, report)


def test_historical_note_reuse_requires_same_cases_and_evidence():
    predictions, examples = fixture_rows()
    selected, _ = select_errors(predictions, examples, .5)
    notes = notes_for(selected)
    historical = attach_reviews(selected, notes)
    assert match_historical_reviews(selected, notes, historical) == historical
    assert match_historical_reviews(selected[:-1], notes, historical) is None
    assert match_historical_reviews(
        [dict(selected[0], reference_context="Different evidence."), *selected[1:]], notes, historical,
    ) is None
