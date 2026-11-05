"""Network-free regression tests for corrected Phase 3 preprocessing."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hallucination_detector.features.preprocessing import normalize_text, PREPROCESSING_VERSION
from hallucination_detector.features.pipeline import (
    _deduplicate_texts, _load_or_compute_embeddings, load_split_examples,
    run_phase3_pipeline,
)
from hallucination_detector.features.embeddings import EmbeddingCache
from hallucination_detector.features.encoder import FrozenEncoder
from hallucination_detector.features.chunking import chunk_context
from hallucination_detector.features.semantic_audit import run_train_semantic_audit
from tests.test_phase3 import FakeEncoder, FakeTokenizer


@pytest.fixture
def tmp_path():
    """Keep atomic cache moves inside the writable repository on Windows."""
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        yield Path(directory)


@pytest.mark.parametrize("text", ["  alpha beta  ", "alpha   beta", "alpha\tbeta",
                                  "alpha\n\nbeta", "\u2003alpha\u00a0\u202fbeta\u3000"])
def test_whitespace_normalization(text):
    assert normalize_text(text) == "alpha beta"
    assert normalize_text(normalize_text(text)) == "alpha beta"


@pytest.mark.parametrize("text", [None, 42, b"text", []])
def test_normalization_requires_string(text):
    with pytest.raises(TypeError, match="string"):
        normalize_text(text)


def test_normalized_deduplication_is_deterministic():
    examples = [{"text": text} for text in [" x\ty ", "x y", "z", "x\u2003y"]]
    expected = (["x y", "z"], [0, 0, 1, 0])
    assert _deduplicate_texts(examples, "text") == expected
    assert _deduplicate_texts(examples, "text") == expected
    assert examples[0]["text"] == " x\ty "


def test_encoder_and_chunker_normalize_at_boundaries():
    encoder = FrozenEncoder.__new__(FrozenEncoder)
    encoder._batch_size = 64
    encoder._model = MagicMock()
    embeddings = np.ones((1, 384), dtype=np.float32) / np.sqrt(384)
    encoder._model.encode.return_value = embeddings
    encoder.encode_texts([" alpha\t beta\n"])
    assert encoder._model.encode.call_args.args[0] == ["alpha beta"]
    tokenizer = MagicMock(wraps=FakeTokenizer())
    assert chunk_context(" alpha\u2003beta ", tokenizer) == ["alpha beta"]
    assert tokenizer.call_args.args[0] == "alpha beta"


def test_old_preprocessing_cache_rejected(tmp_path):
    encoder = FakeEncoder()
    cache = EmbeddingCache(tmp_path, "shared")
    _load_or_compute_embeddings(["hello"], "test", cache, encoder,
                                {"preprocessing_version": "1.0.0"})
    encoder.encode_texts = MagicMock(side_effect=AssertionError("must reject before encoding"))
    with pytest.raises(ValueError, match="Preprocessing version mismatch"):
        _load_or_compute_embeddings(["hello"], "test", cache, encoder,
                                    {"preprocessing_version": PREPROCESSING_VERSION})
    encoder.encode_texts.assert_not_called()


def test_pipeline_normalizes_without_changing_identity(tmp_path):
    from hallucination_detector.features import pipeline

    processed = tmp_path / "processed"
    processed.mkdir()
    originals = {}
    for split in ("train", "validation", "test"):
        originals[split] = [dict(example_id=f"{split}-{label}", source_id=f"source-{split}",
                                 split=split, label=label, question=" Q\n text ",
                                 candidate_answer=" A\t42 " if label == 0 else " B\u2003 7 ",
                                 reference_context=" Context\n A 42 ") for label in (0, 1)]
        (processed / f"halueval_expanded_{split}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in originals[split]), encoding="utf-8")
    before = {p: p.read_bytes() for p in processed.iterdir()}
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"sha256": "fixture"}', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('[]', encoding="utf-8")
    encoder = FakeEncoder()
    original_encode = encoder.encode_texts
    def checked_encode(texts, **kwargs):
        assert all(t == normalize_text(t) for t in texts)
        return original_encode(texts, **kwargs)
    encoder.encode_texts = checked_encode
    with patch.multiple(pipeline, PROCESSED_DIR=processed, PROVENANCE_PATH=provenance,
                        SPLIT_MANIFEST_PATH=manifest, CACHE_DIR=tmp_path / "cache",
                        OUTPUT_DIR=tmp_path / "features", SCALER_DIR=tmp_path / "scaler"):
        for split in originals:
            loaded = load_split_examples(split)
            for old, new in zip(originals[split], loaded):
                for field in ("source_id", "example_id", "split", "label"):
                    assert new[field] == old[field]
                for field in ("question", "candidate_answer", "reference_context"):
                    assert new[field] == normalize_text(old[field])
        result = run_phase3_pipeline(encoder)
    assert all(p.read_bytes() == data for p, data in before.items())
    for split in originals:
        assert result[split]["feature_shape"] == [2, 1543]
        path = tmp_path / "features" / split
        assert json.loads((path / "example_ids.json").read_text()) == [r["example_id"] for r in originals[split]]
        features = np.load(path / "features.npy")
        np.testing.assert_array_equal(features[:, 1540:], [[1., 1., 1.], [0., 0., 1.]])


def test_semantic_audit_requires_only_train_artifacts(tmp_path):
    train = tmp_path / "data/features/train"
    train.mkdir(parents=True)
    features = np.zeros((4, 1543), dtype=np.float32)
    features[:, 1536] = [0.8, 0.2, 0.6, 0.4]
    np.save(train / "features.npy", features)
    np.save(train / "labels.npy", [0, 1, 0, 1])
    schema = {"scalar_features": {"order": [{"name": "max_answer_context_cosine", "dim_index": 1536}]}}
    (train.parent / "schema.json").write_text(json.dumps(schema))
    for name in ("data/raw/qa_data.json", "data/splits/halueval_original_split_manifest.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    # Validation/test files deliberately do not exist.
    report = run_train_semantic_audit(tmp_path)
    assert report["split"] == "train" and report["scaled"] is False
    assert report["statistics"]["grounded"]["count"] == 2
    assert report["statistics"]["grounded"]["mean"] == pytest.approx(0.7)
    assert report["statistics"]["hallucinated"]["mean"] == pytest.approx(0.3)
    assert (tmp_path / "results/figures/train_answer_context_cosine.png").exists()
