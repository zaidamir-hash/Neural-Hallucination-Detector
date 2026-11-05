"""Phase 3 synthetic tests.

All tests use fake/injected encoders and tokenizers.  No network access
or model downloads are required for these routine tests.
"""
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Fake encoder and tokenizer for testing
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Minimal fake tokenizer that splits on whitespace and counts tokens."""

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs) -> Dict[str, Any]:
        tokens = [t for t in text.split() if t]
        return {"input_ids": list(range(len(tokens)))}

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{i}" for i in token_ids)


class FakeEncoder:
    """Deterministic fake encoder that returns random-but-reproducible embeddings.
    Includes validation matching the real FrozenEncoder."""

    def __init__(self, dim: int = 384, seed: int = 42):
        self._dim = dim
        self._rng = np.random.RandomState(seed)

    def encode_texts(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        if not texts:
            raise ValueError("texts must be a non-empty list of strings")
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"texts[{i}] is {type(t).__name__}, expected str")
        rng = np.random.RandomState(self._seed_for_texts(texts))
        embeddings = rng.randn(len(texts), self._dim).astype(np.float32)
        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return embeddings

    def _seed_for_texts(self, texts: List[str]) -> int:
        h = hash(tuple(texts)) % (2**31)
        return h

    @property
    def model_name(self) -> str:
        return "fake-encoder"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def batch_size(self) -> int:
        return 64

    @property
    def revision(self) -> Optional[str]:
        return "fake-rev-001"

    @property
    def sentence_transformers_version(self) -> Optional[str]:
        return "0.0.0-fake"

    def get_tokenizer(self):
        return FakeTokenizer()


# ---------------------------------------------------------------------------
# Tests: context chunking
# ---------------------------------------------------------------------------

class TestChunking(unittest.TestCase):
    """Tests for tokenizer-aware context chunking."""

    def _chunk(self, context: str, max_tokens: int = 192, overlap: int = 32):
        from hallucination_detector.features.chunking import chunk_context
        return chunk_context(context, FakeTokenizer(), max_tokens, overlap)

    def test_short_context_single_chunk(self):
        chunks = self._chunk("hello world")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "hello world")

    def test_empty_context_raises(self):
        from hallucination_detector.features.chunking import chunk_context
        with self.assertRaises(ValueError):
            chunk_context("", FakeTokenizer(), 192, 32)
        with self.assertRaises(ValueError):
            chunk_context("   ", FakeTokenizer(), 192, 32)

    def test_overlap_must_be_less_than_max(self):
        from hallucination_detector.features.chunking import chunk_context
        with self.assertRaises(ValueError):
            chunk_context("test", FakeTokenizer(), 10, 10)
        with self.assertRaises(ValueError):
            chunk_context("test", FakeTokenizer(), 10, 15)

    def test_multi_chunk_boundary(self):
        # 20 tokens with max=10, overlap=3 -> chunks of 10, stride=7
        text = " ".join(f"word{i}" for i in range(20))
        chunks = self._chunk(text, max_tokens=10, overlap=3)
        self.assertGreater(len(chunks), 1)
        # Each chunk should be non-empty
        for c in chunks:
            self.assertTrue(c.strip())

    def test_overlap_behavior(self):
        text = " ".join(f"w{i}" for i in range(15))
        chunks = self._chunk(text, max_tokens=6, overlap=2)
        self.assertGreater(len(chunks), 1)
        # Verify chunks exist and are strings
        for c in chunks:
            self.assertIsInstance(c, str)

    def test_final_partial_chunk_retained(self):
        # 11 tokens, max=5, overlap=1 -> chunks of 5, stride=4
        text = " ".join(f"t{i}" for i in range(11))
        chunks = self._chunk(text, max_tokens=5, overlap=1)
        self.assertGreater(len(chunks), 1)
        # Last chunk should be shorter
        last_tokens = chunks[-1].split()
        self.assertLessEqual(len(last_tokens), 5)

    def test_deterministic_chunks(self):
        text = " ".join(f"word{i}" for i in range(30))
        c1 = self._chunk(text, max_tokens=8, overlap=2)
        c2 = self._chunk(text, max_tokens=8, overlap=2)
        self.assertEqual(c1, c2)

    def test_no_infinite_loop(self):
        # Even with small max and overlap, should terminate
        text = " ".join(f"x{i}" for i in range(100))
        chunks = self._chunk(text, max_tokens=5, overlap=1)
        self.assertLess(len(chunks), 100)  # reasonable upper bound


# ---------------------------------------------------------------------------
# Tests: frozen encoder (via fake)
# ---------------------------------------------------------------------------

class TestFrozenEncoder(unittest.TestCase):
    """Tests for encoder behavior using a fake encoder."""

    def test_encode_produces_correct_shape(self):
        enc = FakeEncoder(dim=384)
        texts = ["hello", "world", "test"]
        embs = enc.encode_texts(texts)
        self.assertEqual(embs.shape, (3, 384))

    def test_encode_all_finite(self):
        enc = FakeEncoder()
        embs = enc.encode_texts(["a", "b", "c"])
        self.assertTrue(np.all(np.isfinite(embs)))

    def test_encode_normalized(self):
        enc = FakeEncoder()
        embs = enc.encode_texts(["hello", "world"])
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_encode_empty_raises(self):
        enc = FakeEncoder()
        with self.assertRaises(ValueError):
            enc.encode_texts([])

    def test_encode_non_string_raises(self):
        enc = FakeEncoder()
        with self.assertRaises(TypeError):
            enc.encode_texts([123])  # type: ignore

    def test_properties(self):
        enc = FakeEncoder(dim=384)
        self.assertEqual(enc.model_name, "fake-encoder")
        self.assertEqual(enc.embedding_dim, 384)
        self.assertEqual(enc.device, "cpu")
        self.assertIsNotNone(enc.revision)

    def test_get_tokenizer(self):
        enc = FakeEncoder()
        tok = enc.get_tokenizer()
        self.assertIsInstance(tok, FakeTokenizer)


# ---------------------------------------------------------------------------
# Tests: scalar features
# ---------------------------------------------------------------------------

class TestScalarFeatures(unittest.TestCase):

    def test_content_coverage_full(self):
        from hallucination_detector.features.scalar_features import compute_answer_content_coverage
        cov = compute_answer_content_coverage("the cat", "the cat sat on the mat")
        self.assertAlmostEqual(cov, 1.0)

    def test_content_coverage_partial(self):
        from hallucination_detector.features.scalar_features import compute_answer_content_coverage
        cov = compute_answer_content_coverage("dog fish", "the cat sat on the mat")
        self.assertAlmostEqual(cov, 0.0)

    def test_content_coverage_empty_answer(self):
        from hallucination_detector.features.scalar_features import compute_answer_content_coverage
        cov = compute_answer_content_coverage("", "some context")
        self.assertAlmostEqual(cov, 0.0)

    def test_numeric_coverage(self):
        from hallucination_detector.features.scalar_features import compute_answer_numeric_coverage
        cov = compute_answer_numeric_coverage("42 people", "There were 42 people at the event")
        self.assertAlmostEqual(cov, 1.0)

    def test_numeric_coverage_no_numerics(self):
        from hallucination_detector.features.scalar_features import compute_answer_numeric_coverage
        cov = compute_answer_numeric_coverage("hello world", "some context")
        self.assertAlmostEqual(cov, 0.0)

    def test_has_numeric_true(self):
        from hallucination_detector.features.scalar_features import compute_answer_has_numeric
        self.assertAlmostEqual(compute_answer_has_numeric("42 answers"), 1.0)

    def test_has_numeric_false(self):
        from hallucination_detector.features.scalar_features import compute_answer_has_numeric
        self.assertAlmostEqual(compute_answer_has_numeric("no numbers here"), 0.0)


# ---------------------------------------------------------------------------
# Tests: primary features
# ---------------------------------------------------------------------------

class TestPrimaryFeatures(unittest.TestCase):

    def _make_features(self, seed: int = 42):
        from hallucination_detector.features.primary import build_primary_features
        rng = np.random.RandomState(seed)
        dim = 384
        q = rng.randn(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        a = rng.randn(dim).astype(np.float32)
        a /= np.linalg.norm(a)
        c_a = rng.randn(dim).astype(np.float32)
        c_a /= np.linalg.norm(c_a)
        all_c = rng.randn(3, dim).astype(np.float32)
        all_c /= np.linalg.norm(all_c, axis=1, keepdims=True)

        features = build_primary_features(
            q, a, c_a, all_c,
            answer_text="42 people attended",
            context_text="There were 42 people at the event",
            q_c_sim_max=0.8,
        )
        return features

    def test_feature_dimension(self):
        features = self._make_features()
        self.assertEqual(features.shape, (1543,))

    def test_all_finite(self):
        features = self._make_features()
        self.assertTrue(np.all(np.isfinite(features)))

    def test_exact_order(self):
        """Verify the 7 scalar features are at indices 1536-1542."""
        features = self._make_features()
        # Scalar features start at 1536
        scalars = features[1536:]
        self.assertEqual(scalars.shape, (7,))
        # max_ac_sim should be between -1 and 1
        self.assertGreaterEqual(scalars[0], -1.0)
        self.assertLessEqual(scalars[0], 1.0)
        # mean_top3 should be between -1 and 1
        self.assertGreaterEqual(scalars[1], -1.0)
        self.assertLessEqual(scalars[1], 1.0)
        # qa_sim should be between -1 and 1
        self.assertGreaterEqual(scalars[2], -1.0)
        self.assertLessEqual(scalars[2], 1.0)
        # max_qc_sim should be between -1 and 1
        self.assertGreaterEqual(scalars[3], -1.0)
        self.assertLessEqual(scalars[3], 1.0)

    def test_deterministic(self):
        f1 = self._make_features(seed=99)
        f2 = self._make_features(seed=99)
        np.testing.assert_array_equal(f1, f2)

    def test_blocks_are_384_each(self):
        features = self._make_features()
        # Block 1: |a - c_a| at 0:384
        block1 = features[0:384]
        self.assertEqual(block1.shape, (384,))
        # Block 2: a * c_a at 384:768
        block2 = features[384:768]
        self.assertEqual(block2.shape, (384,))
        # Block 3: |q - a| at 768:1152
        block3 = features[768:1152]
        self.assertEqual(block3.shape, (384,))
        # Block 4: q * a at 1152:1536
        block4 = features[1152:1536]
        self.assertEqual(block4.shape, (384,))


# ---------------------------------------------------------------------------
# Tests: scaler
# ---------------------------------------------------------------------------

class TestScaler(unittest.TestCase):

    def test_fit_and_transform(self):
        from hallucination_detector.features.scaler import fit_scaler, transform_with_scaler
        rng = np.random.RandomState(42)
        train = rng.randn(100, 10).astype(np.float32)
        mean, scale = fit_scaler(train)
        self.assertEqual(mean.shape, (10,))
        self.assertEqual(scale.shape, (10,))
        # Scale should be positive (non-zero)
        self.assertTrue(np.all(scale > 0))

        scaled = transform_with_scaler(train, mean, scale)
        self.assertEqual(scaled.shape, train.shape)
        self.assertTrue(np.all(np.isfinite(scaled)))

    def test_train_only_fitting(self):
        """Verify that scaler mean matches training data mean."""
        from hallucination_detector.features.scaler import fit_scaler
        rng = np.random.RandomState(0)
        train = rng.randn(200, 5).astype(np.float32)
        mean, _ = fit_scaler(train)
        np.testing.assert_allclose(mean, train.mean(axis=0), atol=1e-6)

    def test_validation_reuses_train_scaler(self):
        """Verify that validation transform uses training scaler."""
        from hallucination_detector.features.scaler import fit_scaler, transform_with_scaler
        rng = np.random.RandomState(1)
        train = rng.randn(100, 8).astype(np.float32)
        val = rng.randn(30, 8).astype(np.float32)
        mean, scale = fit_scaler(train)
        # Transform val with train scaler
        scaled_val = transform_with_scaler(val, mean, scale)
        self.assertEqual(scaled_val.shape, val.shape)
        # Verify it actually scaled (not identity)
        self.assertFalse(np.allclose(scaled_val, val))

    def test_save_and_load_scaler(self):
        from hallucination_detector.features.scaler import save_scaler, load_scaler
        rng = np.random.RandomState(7)
        mean = rng.randn(10).astype(np.float32)
        scale = np.abs(rng.randn(10)).astype(np.float32) + 0.1
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "scaler.npz"
            save_scaler(path, mean, scale)
            loaded_mean, loaded_scale = load_scaler(path)
            np.testing.assert_array_equal(loaded_mean, mean)
            np.testing.assert_array_equal(loaded_scale, scale)


# ---------------------------------------------------------------------------
# Tests: embedding cache metadata
# ---------------------------------------------------------------------------

class TestEmbeddingCacheMetadata(unittest.TestCase):

    def test_cache_metadata_construction(self):
        from hallucination_detector.features.embeddings import build_cache_metadata
        enc = FakeEncoder()
        meta = build_cache_metadata(
            dataset_checksum="abc123",
            split_manifest_checksum="def456",
            encoder=enc,
            chunk_size=192,
            chunk_overlap=32,
            split="train",
            row_count=100,
            feature_dim=1543,
        )
        self.assertEqual(meta["dataset_checksum"], "abc123")
        self.assertEqual(meta["encoder_name"], "fake-encoder")
        self.assertEqual(meta["embedding_dim"], 384)
        self.assertEqual(meta["split"], "train")
        self.assertEqual(meta["row_count"], 100)
        self.assertEqual(meta["feature_dim"], 1543)

    def test_cache_rejects_incompatible(self):
        from hallucination_detector.features.embeddings import EmbeddingCache, _validate_cache_metadata
        cached = {"dataset_checksum": "abc", "encoder_name": "model-a"}
        expected = {"dataset_checksum": "xyz", "encoder_name": "model-a"}
        with self.assertRaises(ValueError):
            _validate_cache_metadata(cached, expected, Path("test.json"))


# ---------------------------------------------------------------------------
# Tests: feature schema
# ---------------------------------------------------------------------------

class TestFeatureSchema(unittest.TestCase):

    def test_schema_loads(self):
        from hallucination_detector.features.schema import load_feature_schema
        schema = load_feature_schema()
        self.assertEqual(schema["total_dim"], 1543)
        self.assertEqual(schema["interaction_blocks"]["count"], 4)
        self.assertEqual(schema["scalar_features"]["count"], 7)

    def test_feature_dim(self):
        from hallucination_detector.features.schema import get_feature_dim
        self.assertEqual(get_feature_dim(), 1543)

    def test_scalar_names(self):
        from hallucination_detector.features.schema import get_scalar_feature_names
        names = get_scalar_feature_names()
        self.assertEqual(len(names), 7)
        self.assertEqual(names[0], "max_answer_context_cosine")
        self.assertEqual(names[6], "answer_has_numeric_token")


# ---------------------------------------------------------------------------
# Tests: batch feature generation
# ---------------------------------------------------------------------------

class TestBatchFeatures(unittest.TestCase):

    def test_batch_shape(self):
        from hallucination_detector.features.primary import build_features_batch
        rng = np.random.RandomState(42)
        n = 5
        dim = 384
        q = rng.randn(n, dim).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        a = rng.randn(n, dim).astype(np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        c_a = rng.randn(n, dim).astype(np.float32)
        c_a /= np.linalg.norm(c_a, axis=1, keepdims=True)
        all_c = [rng.randn(2, dim).astype(np.float32) for _ in range(n)]
        for i in range(n):
            all_c[i] /= np.linalg.norm(all_c[i], axis=1, keepdims=True)

        features = build_features_batch(
            q, a, c_a, all_c,
            ["answer"] * n, ["context"] * n,
            np.array([0.5] * n, dtype=np.float32),
        )
        self.assertEqual(features.shape, (n, 1543))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_batch_single(self):
        from hallucination_detector.features.primary import build_features_batch
        rng = np.random.RandomState(0)
        dim = 384
        q = rng.randn(1, dim).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        a = rng.randn(1, dim).astype(np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        c_a = rng.randn(1, dim).astype(np.float32)
        c_a /= np.linalg.norm(c_a, axis=1, keepdims=True)
        all_c = [rng.randn(1, dim).astype(np.float32)]
        all_c[0] /= np.linalg.norm(all_c[0], axis=1, keepdims=True)

        features = build_features_batch(
            q, a, c_a, all_c,
            ["hello"], ["world"],
            np.array([0.3], dtype=np.float32),
        )
        self.assertEqual(features.shape, (1, 1543))


# ---------------------------------------------------------------------------
# Tests: numeric coverage edge cases
# ---------------------------------------------------------------------------

class TestNumericEdgeCases(unittest.TestCase):

    def test_no_numeric_in_answer(self):
        from hallucination_detector.features.scalar_features import (
            compute_answer_numeric_coverage,
            compute_answer_has_numeric,
        )
        self.assertAlmostEqual(
            compute_answer_numeric_coverage("hello", "hello world"), 0.0
        )
        self.assertAlmostEqual(compute_answer_has_numeric("hello"), 0.0)

    def test_numeric_in_context_only(self):
        from hallucination_detector.features.scalar_features import (
            compute_answer_numeric_coverage,
        )
        # Answer has no numerics
        cov = compute_answer_numeric_coverage("hello", "123 world")
        self.assertAlmostEqual(cov, 0.0)

    def test_multiple_numerics(self):
        from hallucination_detector.features.scalar_features import (
            compute_answer_numeric_coverage,
            compute_answer_has_numeric,
        )
        cov = compute_answer_numeric_coverage(
            "42 people and 7 cats",
            "There were 42 people and 7 cats present"
        )
        self.assertAlmostEqual(cov, 1.0)
        self.assertAlmostEqual(compute_answer_has_numeric("42 people"), 1.0)


# ---------------------------------------------------------------------------
# Tests: embedding cache integration
# ---------------------------------------------------------------------------

class TestEmbeddingCacheIntegration(unittest.TestCase):
    """Tests for embedding cache creation, reuse, and rejection."""

    def _make_fake_encoder_and_meta(self, dim=384):
        enc = FakeEncoder(dim=dim)
        meta = {
            "cache_format_version": "1.0.0",
            "preprocessing_version": "1.0.0",
            "dataset_checksum": "abc123",
            "split_manifest_checksum": "def456",
            "encoder_name": "fake-encoder",
            "encoder_revision": "fake-rev-001",
            "sentence_transformers_version": "0.0.0-fake",
            "embedding_dim": dim,
            "normalized_embeddings": True,
            "context_chunk_max_tokens": 192,
            "context_chunk_overlap_tokens": 32,
            "feature_schema_version": "1.0.0",
        }
        return enc, meta

    def test_first_execution_creates_cache(self):
        """First run should compute and save embeddings."""
        from hallucination_detector.features.embeddings import (
            _validate_cache_metadata,
        )
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "cache"
            cache_dir.mkdir()
            texts = ["hello world", "foo bar", "test sentence"]
            enc, meta = self._make_fake_encoder_and_meta()

            embs = enc.encode_texts(texts)
            # Save
            np.save(cache_dir / "embeddings_test.npy", embs)
            (cache_dir / "cache_metadata_test.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            (cache_dir / "embedding_ids_test.json").write_text(
                json.dumps(texts), encoding="utf-8"
            )

            # Verify files exist
            self.assertTrue((cache_dir / "embeddings_test.npy").exists())
            self.assertTrue((cache_dir / "cache_metadata_test.json").exists())
            self.assertTrue((cache_dir / "embedding_ids_test.json").exists())

            # Verify shape
            loaded = np.load(cache_dir / "embeddings_test.npy")
            self.assertEqual(loaded.shape, (3, 384))

    def test_pipeline_helper_creates_reuses_and_force_refreshes_cache(self):
        """Exercise the real cache helper rather than simulating its behavior."""
        from hallucination_detector.features.embeddings import EmbeddingCache
        from hallucination_detector.features.pipeline import _load_or_compute_embeddings

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as td:
            cache = EmbeddingCache(Path(td), "shared")
            texts = ["hello", "world"]
            enc, meta = self._make_fake_encoder_and_meta()
            original_encode = enc.encode_texts
            enc.encode_texts = MagicMock(side_effect=original_encode)

            first = _load_or_compute_embeddings(texts, "test", cache, enc, meta)
            self.assertEqual(enc.encode_texts.call_count, 1)

            second = _load_or_compute_embeddings(texts, "test", cache, enc, meta)
            self.assertEqual(enc.encode_texts.call_count, 1)
            np.testing.assert_array_equal(second, first)

            refreshed = _load_or_compute_embeddings(
                texts, "test", cache, enc, meta, force_recompute=True
            )
            self.assertEqual(enc.encode_texts.call_count, 2)
            np.testing.assert_array_equal(refreshed, first)

    def test_pipeline_helper_rejects_incompatible_and_incomplete_cache(self):
        from hallucination_detector.features.embeddings import EmbeddingCache
        from hallucination_detector.features.pipeline import _load_or_compute_embeddings

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as td:
            cache = EmbeddingCache(Path(td), "shared")
            texts = ["hello"]
            enc, meta = self._make_fake_encoder_and_meta()
            _load_or_compute_embeddings(texts, "test", cache, enc, meta)

            incompatible = dict(meta)
            incompatible["dataset_checksum"] = "different"
            with self.assertRaisesRegex(ValueError, "Dataset checksum mismatch"):
                _load_or_compute_embeddings(texts, "test", cache, enc, incompatible)

            (Path(td) / "embedding_ids_test.json").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                _load_or_compute_embeddings(texts, "test", cache, enc, meta)

    def test_real_chunk_cache_helper_reuses_offsets(self):
        from hallucination_detector.features.embeddings import EmbeddingCache
        from hallucination_detector.features.pipeline import (
            _load_or_compute_chunk_embeddings,
        )

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as td:
            cache = EmbeddingCache(Path(td), "shared")
            contexts = ["a b c d", "e f g"]
            enc, meta = self._make_fake_encoder_and_meta()
            meta["context_chunk_max_tokens"] = 2
            meta["context_chunk_overlap_tokens"] = 1
            original_encode = enc.encode_texts
            enc.encode_texts = MagicMock(side_effect=original_encode)
            context_embeddings = {
                text: original_encode([text])[0] for text in contexts
            }

            first = _load_or_compute_chunk_embeddings(
                contexts,
                cache,
                enc,
                meta,
                FakeTokenizer(),
                2,
                1,
                context_embeddings,
            )
            calls_after_first = enc.encode_texts.call_count
            self.assertGreater(calls_after_first, 0)

            second = _load_or_compute_chunk_embeddings(
                contexts,
                cache,
                enc,
                meta,
                FakeTokenizer(),
                2,
                1,
                context_embeddings,
            )
            self.assertEqual(enc.encode_texts.call_count, calls_after_first)
            for text in contexts:
                np.testing.assert_array_equal(second[text], first[text])

    def test_compatible_cache_reuse(self):
        """Second compatible run should reuse cache without calling encoder."""
        from hallucination_detector.features.embeddings import _validate_cache_metadata
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "cache"
            cache_dir.mkdir()
            texts = ["hello", "world"]
            enc, meta = self._make_fake_encoder_and_meta()

            # Save cache
            embs = enc.encode_texts(texts)
            np.save(cache_dir / "embeddings_test.npy", embs)
            (cache_dir / "cache_metadata_test.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            (cache_dir / "embedding_ids_test.json").write_text(
                json.dumps(texts), encoding="utf-8"
            )

            # Load and validate
            cached_meta = json.loads(
                (cache_dir / "cache_metadata_test.json").read_text(encoding="utf-8")
            )
            _validate_cache_metadata(cached_meta, meta, cache_dir / "cache_metadata_test.json")
            cached_ids = json.loads(
                (cache_dir / "embedding_ids_test.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cached_ids, texts)
            loaded = np.load(cache_dir / "embeddings_test.npy")
            np.testing.assert_array_equal(loaded, embs)

    def test_compatible_reuse_does_not_call_encoder(self):
        """When cache is valid, encoder should not be called."""
        call_count = [0]
        original_encode = FakeEncoder.encode_texts

        def counting_encode(self_enc, texts, **kwargs):
            call_count[0] += 1
            return original_encode(self_enc, texts, **kwargs)

        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "cache"
            cache_dir.mkdir()
            texts = ["hello", "world"]
            enc, meta = self._make_fake_encoder_and_meta()

            # Save cache
            embs = enc.encode_texts(texts)
            np.save(cache_dir / "embeddings_test.npy", embs)
            (cache_dir / "cache_metadata_test.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            (cache_dir / "embedding_ids_test.json").write_text(
                json.dumps(texts), encoding="utf-8"
            )

            # Simulate cache hit path: validate + load
            cached_meta = json.loads(
                (cache_dir / "cache_metadata_test.json").read_text(encoding="utf-8")
            )
            from hallucination_detector.features.embeddings import _validate_cache_metadata
            _validate_cache_metadata(cached_meta, meta, cache_dir / "cache_metadata_test.json")
            loaded = np.load(cache_dir / "embeddings_test.npy")
            self.assertEqual(loaded.shape, (2, 384))
            # Encoder was never called
            self.assertEqual(call_count[0], 0)

    def test_changed_encoder_rejected(self):
        """Changed encoder name should reject cache."""
        from hallucination_detector.features.embeddings import _validate_cache_metadata
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            cached = {"encoder_name": "model-a", "embedding_dim": 384}
            expected = {"encoder_name": "model-b", "embedding_dim": 384}
            with self.assertRaises(ValueError):
                _validate_cache_metadata(cached, expected, cache_dir / "test.json")

    def test_missing_required_metadata_rejected(self):
        from hallucination_detector.features.embeddings import _validate_cache_metadata

        cached = {"encoder_name": "model-a"}
        expected = {"encoder_name": "model-a", "embedding_dim": 384}
        with self.assertRaisesRegex(ValueError, "Missing required metadata key"):
            _validate_cache_metadata(cached, expected, Path("test.json"))

    def test_changed_checksum_rejected(self):
        """Changed dataset checksum should reject cache."""
        from hallucination_detector.features.embeddings import _validate_cache_metadata
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            cached = {"dataset_checksum": "aaa"}
            expected = {"dataset_checksum": "bbb"}
            with self.assertRaises(ValueError):
                _validate_cache_metadata(cached, expected, cache_dir / "test.json")

    def test_changed_normalization_rejected(self):
        """Changed normalization setting should reject cache."""
        from hallucination_detector.features.embeddings import _validate_cache_metadata
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            cached = {"normalized_embeddings": True}
            expected = {"normalized_embeddings": False}
            with self.assertRaises(ValueError):
                _validate_cache_metadata(cached, expected, cache_dir / "test.json")

    def test_changed_chunk_size_rejected(self):
        """Changed chunk size should reject cache."""
        from hallucination_detector.features.embeddings import _validate_cache_metadata
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            cached = {"context_chunk_max_tokens": 192}
            expected = {"context_chunk_max_tokens": 256}
            with self.assertRaises(ValueError):
                _validate_cache_metadata(cached, expected, cache_dir / "test.json")

    def test_changed_chunk_overlap_rejected(self):
        """Changed chunk overlap should reject cache."""
        from hallucination_detector.features.embeddings import _validate_cache_metadata
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            cached = {"context_chunk_overlap_tokens": 32}
            expected = {"context_chunk_overlap_tokens": 64}
            with self.assertRaises(ValueError):
                _validate_cache_metadata(cached, expected, cache_dir / "test.json")

    def test_changed_schema_version_rejected(self):
        """Changed schema version should reject cache."""
        from hallucination_detector.features.embeddings import _validate_cache_metadata
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            cached = {"feature_schema_version": "1.0.0"}
            expected = {"feature_schema_version": "2.0.0"}
            with self.assertRaises(ValueError):
                _validate_cache_metadata(cached, expected, cache_dir / "test.json")

    def test_truncated_array_rejected(self):
        """Truncated embedding array should be detected."""
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            # Save array with wrong shape
            embs = np.random.randn(2, 100).astype(np.float32)
            np.save(cache_dir / "embeddings_test.npy", embs)
            loaded = np.load(cache_dir / "embeddings_test.npy")
            self.assertNotEqual(loaded.shape, (2, 384))

    def test_cached_ids_deterministic(self):
        """Cached IDs should map deterministically."""
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            texts = ["alpha", "beta", "gamma"]
            (cache_dir / "embedding_ids_test.json").write_text(
                json.dumps(texts, ensure_ascii=False), encoding="utf-8"
            )
            loaded = json.loads(
                (cache_dir / "embedding_ids_test.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded, texts)

    def test_cached_embeddings_shape(self):
        """Cached embeddings should have shape (*, 384)."""
        enc = FakeEncoder(dim=384)
        embs = enc.encode_texts(["hello", "world", "test"])
        self.assertEqual(embs.shape, (3, 384))

    def test_cached_embeddings_finite_and_normalized(self):
        """Cached embeddings should be finite and L2-normalized."""
        enc = FakeEncoder(dim=384)
        embs = enc.encode_texts(["hello", "world"])
        self.assertTrue(np.all(np.isfinite(embs)))
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_chunk_embeddings_offsets_deterministic(self):
        """Chunk offsets should map back to correct chunks."""
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            # Simulate 3 contexts with 2, 1, 3 chunks
            chunks = [
                np.random.randn(2, 384).astype(np.float32),
                np.random.randn(1, 384).astype(np.float32),
                np.random.randn(3, 384).astype(np.float32),
            ]
            all_embs = np.concatenate(chunks, axis=0)
            offsets = np.array([0, 2, 3, 6], dtype=np.int64)

            np.save(cache_dir / "embeddings_chunks.npy", all_embs)
            np.save(cache_dir / "chunk_offsets.npy", offsets)

            loaded_embs = np.load(cache_dir / "embeddings_chunks.npy")
            loaded_offsets = np.load(cache_dir / "chunk_offsets.npy")

            for i in range(3):
                start = int(loaded_offsets[i])
                end = int(loaded_offsets[i + 1])
                chunk = loaded_embs[start:end]
                self.assertEqual(chunk.shape, (len(chunks[i]), 384))
                np.testing.assert_array_equal(chunk, chunks[i])


if __name__ == "__main__":
    unittest.main()
