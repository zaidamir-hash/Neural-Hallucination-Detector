"""Phase 3 pipeline orchestrator.

Coordinates loading Phase 2 data, encoding, chunking, feature generation,
scaling, and caching.  This module does not train classifiers or evaluate
test performance.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hallucination_detector.config import load_config
from hallucination_detector.features.encoder import FrozenEncoder, EMBEDDING_DIM
from hallucination_detector.features.chunking import chunk_context
from hallucination_detector.features.primary import build_features_batch
from hallucination_detector.features.scaler import (
    fit_scaler,
    transform_with_scaler,
    save_scaler,
    load_scaler,
)
from hallucination_detector.features.embeddings import (
    EmbeddingCache,
    build_cache_metadata,
    _validate_cache_metadata,
    PREPROCESSING_VERSION,
    CACHE_FORMAT_VERSION,
)
from hallucination_detector.features.schema import load_feature_schema

logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/qa_data.json")
PHASE1_MANIFEST = Path("data/metadata/halueval_manifest.json")
PHASE1_VALIDATION_REPORT = Path("data/metadata/halueval_validation_report.json")
SPLIT_MANIFEST_PATH = Path("data/splits/halueval_original_split_manifest.json")
PROVENANCE_PATH = Path("data/metadata/halueval_qa_provenance.json")
PROCESSED_DIR = Path("data/processed")
CACHE_DIR = Path("data/embedding_cache")
SCALER_DIR = Path("data/scaler")
OUTPUT_DIR = Path("data/features")
SCHEMA_PATH = Path("configs/feature_schema.json")


def _replace_cache_file(temp_path: str, destination: Path) -> None:
    """Replace a cache file, with a Windows/OneDrive-compatible fallback."""
    import os

    try:
        os.replace(temp_path, str(destination))
    except PermissionError:
        # Some Windows/OneDrive configurations reject replacement of an
        # existing file. Force refresh is explicit, so remove only the known
        # cache destination before retrying the same-directory move.
        destination.unlink(missing_ok=True)
        os.replace(temp_path, str(destination))


def _compute_file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_split_examples(split: str) -> List[Dict[str, Any]]:
    """Load expanded examples for a split from Phase 2 processed data."""
    path = PROCESSED_DIR / f"halueval_expanded_{split}.jsonl"
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def _deduplicate_texts(
    examples: List[Dict[str, Any]], field: str
) -> Tuple[List[str], List[int]]:
    """Extract unique texts for a given field and mapping indices."""
    seen: Dict[str, int] = {}
    unique: List[str] = []
    index_map: List[int] = []
    for ex in examples:
        text = ex[field]
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h not in seen:
            seen[h] = len(unique)
            unique.append(text)
        index_map.append(seen[h])
    return unique, index_map


def _build_expected_cache_meta(
    dataset_checksum: str,
    split_manifest_checksum: str,
    encoder: FrozenEncoder,
    chunk_size: int,
    chunk_overlap: int,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the expected metadata dict for cache validation."""
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset_checksum": dataset_checksum,
        "split_manifest_checksum": split_manifest_checksum,
        "encoder_name": encoder.model_name,
        "encoder_revision": encoder.revision,
        "sentence_transformers_version": encoder.sentence_transformers_version,
        "embedding_dim": encoder.embedding_dim,
        "normalized_embeddings": True,
        "context_chunk_max_tokens": chunk_size,
        "context_chunk_overlap_tokens": chunk_overlap,
        "feature_schema_version": schema.get("schema_version", "1.0.0"),
    }


def _load_or_compute_embeddings(
    texts: List[str],
    variant: str,
    cache: EmbeddingCache,
    encoder: FrozenEncoder,
    expected_meta: Dict[str, Any],
    force_recompute: bool = False,
) -> np.ndarray:
    """Load compatible embeddings or explicitly compute and refresh them."""
    meta_path = cache.cache_dir / f"cache_metadata_{variant}.json"
    emb_path = cache.cache_dir / f"embeddings_{variant}.npy"
    ids_path = cache.cache_dir / f"embedding_ids_{variant}.json"

    cache_paths = (meta_path, emb_path, ids_path)
    present_count = sum(path.exists() for path in cache_paths)
    if not force_recompute and present_count:
        if present_count != len(cache_paths):
            raise ValueError(
                f"Incomplete {variant} embedding cache in {cache.cache_dir}; "
                "rerun with --force-recompute to replace it"
            )
        try:
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid cache metadata in {meta_path}; rerun with --force-recompute"
            ) from exc
        _validate_cache_metadata(cached_meta, expected_meta, meta_path)
        cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if cached_ids != texts:
            raise ValueError(
                f"Cached {variant} identifiers do not match current inputs; "
                "rerun with --force-recompute"
            )
        embs = np.load(emb_path, allow_pickle=False)
        expected_shape = (len(texts), expected_meta["embedding_dim"])
        if embs.shape != expected_shape:
            raise ValueError(
                f"Cached {variant} embeddings have shape {embs.shape}, expected "
                f"{expected_shape}; rerun with --force-recompute"
            )
        if not np.all(np.isfinite(embs)):
            raise ValueError(f"Cached {variant} embeddings contain non-finite values")
        norms = np.linalg.norm(embs, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(f"Cached {variant} embeddings are not L2-normalized")
        logger.info("Cache hit for %s (%d embeddings)", variant, len(texts))
        return embs

    # Compute
    logger.info("Encoding %d unique %s...", len(texts), variant)
    embs = encoder.encode_texts(texts, show_progress=True)

    # Save atomically using tempfile
    import tempfile, os
    fd, tmp_path = tempfile.mkstemp(suffix=".npy", dir=str(cache.cache_dir))
    os.close(fd)
    try:
        np.save(tmp_path, embs)
        _replace_cache_file(tmp_path, emb_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    meta_to_save = dict(expected_meta)
    meta_to_save["count"] = len(texts)
    fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=str(cache.cache_dir))
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta_to_save, f, indent=2, ensure_ascii=False)
        _replace_cache_file(tmp_path, meta_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=str(cache.cache_dir))
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)
        _replace_cache_file(tmp_path, ids_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return embs


def _load_or_compute_chunk_embeddings(
    unique_contexts: List[str],
    cache: EmbeddingCache,
    encoder: FrozenEncoder,
    expected_meta: Dict[str, Any],
    tokenizer: Any,
    max_chunk_tokens: int,
    chunk_overlap: int,
    c_emb_map: Dict[str, np.ndarray],
    force_recompute: bool = False,
) -> Dict[str, np.ndarray]:
    """Load chunk embeddings from cache or compute and save them."""
    meta_path = cache.cache_dir / "cache_metadata_chunks.json"
    chunks_npy_path = cache.cache_dir / "embeddings_chunks.npy"
    offsets_path = cache.cache_dir / "chunk_offsets.npy"
    ids_path = cache.cache_dir / "embedding_ids_chunks.json"

    cache_paths = (meta_path, chunks_npy_path, offsets_path, ids_path)
    present_count = sum(path.exists() for path in cache_paths)
    if not force_recompute and present_count:
        if present_count != len(cache_paths):
            raise ValueError(
                f"Incomplete context-chunk cache in {cache.cache_dir}; "
                "rerun with --force-recompute to replace it"
            )
        try:
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid cache metadata in {meta_path}; rerun with --force-recompute"
            ) from exc
        _validate_cache_metadata(cached_meta, expected_meta, meta_path)
        cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if cached_ids != unique_contexts:
            raise ValueError(
                "Cached context identifiers do not match current inputs; "
                "rerun with --force-recompute"
            )
        all_embs = np.load(chunks_npy_path, allow_pickle=False)
        offsets = np.load(offsets_path, allow_pickle=False)
        expected_dim = expected_meta["embedding_dim"]
        if all_embs.ndim != 2 or all_embs.shape[1] != expected_dim:
            raise ValueError(
                f"Cached chunk embeddings have invalid shape {all_embs.shape}"
            )
        if offsets.shape != (len(unique_contexts) + 1,):
            raise ValueError(f"Cached chunk offsets have invalid shape {offsets.shape}")
        if offsets[0] != 0 or offsets[-1] != len(all_embs) or np.any(np.diff(offsets) <= 0):
            raise ValueError("Cached chunk offsets are inconsistent with embeddings")
        if not np.all(np.isfinite(all_embs)):
            raise ValueError("Cached chunk embeddings contain non-finite values")
        if not np.allclose(np.linalg.norm(all_embs, axis=1), 1.0, atol=1e-4):
            raise ValueError("Cached chunk embeddings are not L2-normalized")
        logger.info(
            "Cache hit for context chunks (%d contexts, %d total chunks)",
            len(unique_contexts), len(all_embs),
        )
        result: Dict[str, np.ndarray] = {}
        for i, ctx in enumerate(unique_contexts):
            start, end = int(offsets[i]), int(offsets[i + 1])
            result[ctx] = all_embs[start:end]
        return result

    # Compute
    logger.info("Chunking and encoding context chunks...")
    result = {}
    all_embs_list: List[np.ndarray] = []
    offsets = [0]
    for ctx_text in unique_contexts:
        chunks = chunk_context(ctx_text, tokenizer, max_chunk_tokens, chunk_overlap)
        if len(chunks) == 1:
            chunk_embs = c_emb_map[ctx_text].reshape(1, -1)
        else:
            chunk_embs = encoder.encode_texts(chunks)
        result[ctx_text] = chunk_embs
        all_embs_list.append(chunk_embs)
        offsets.append(offsets[-1] + len(chunk_embs))

    all_embs = np.concatenate(all_embs_list, axis=0)
    offsets_arr = np.array(offsets, dtype=np.int64)

    # Save atomically
    import tempfile, os
    for arr_path, arr_data in [(chunks_npy_path, all_embs), (offsets_path, offsets_arr)]:
        fd, tmp_path = tempfile.mkstemp(suffix=".npy", dir=str(cache.cache_dir))
        os.close(fd)
        try:
            np.save(tmp_path, arr_data)
            _replace_cache_file(tmp_path, arr_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    meta_to_save = dict(expected_meta)
    meta_to_save["total_chunks"] = len(all_embs)
    meta_to_save["unique_contexts"] = len(unique_contexts)
    for jpath, jdata in [(meta_path, meta_to_save), (ids_path, unique_contexts)]:
        fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=str(cache.cache_dir))
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(jdata, f, indent=2, ensure_ascii=False) if isinstance(jdata, dict) else json.dump(jdata, f, ensure_ascii=False)
            _replace_cache_file(tmp_path, jpath)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    return result


def run_phase3_pipeline(
    encoder: Optional[FrozenEncoder] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Run the complete Phase 3 pipeline.

    Steps:
      1. Load config and Phase 2 data
      2. Load or compute embeddings (with cache)
      3. Build primary features
      4. Fit scaler on training data
      5. Transform all splits
      6. Save artifacts

    Args:
        encoder: Optional pre-loaded encoder.
        force_recompute: If True, ignore caches and recompute everything.

    Returns:
        Summary dict with shapes, checksums, timings.
    """
    t0 = time.time()
    cfg = load_config()
    schema = load_feature_schema()
    chunk_cfg = cfg["context_chunking"]
    max_chunk_tokens = chunk_cfg["max_chunk_tokens"]
    chunk_overlap = chunk_cfg["chunk_overlap_tokens"]

    # Load provenance for checksums
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    dataset_checksum = provenance["sha256"]
    split_manifest_checksum = _compute_file_checksum(SPLIT_MANIFEST_PATH)

    if encoder is None:
        encoder = FrozenEncoder()

    tokenizer = encoder.get_tokenizer()

    # Expected cache metadata for validation
    expected_meta = _build_expected_cache_meta(
        dataset_checksum, split_manifest_checksum, encoder,
        max_chunk_tokens, chunk_overlap, schema,
    )

    # Shared cache for question/answer embeddings (not split-specific)
    shared_cache = EmbeddingCache(CACHE_DIR, "shared")

    # Load all split data
    split_data: Dict[str, List[Dict[str, Any]]] = {}
    all_questions_set: set = set()
    all_answers_set: set = set()
    all_contexts_set: set = set()
    for split in ("train", "validation", "test"):
        examples = load_split_examples(split)
        split_data[split] = examples
        for ex in examples:
            all_questions_set.add(ex["question"])
            all_answers_set.add(ex["candidate_answer"])
            all_contexts_set.add(ex["reference_context"])

    unique_questions = sorted(all_questions_set)
    unique_answers = sorted(all_answers_set)
    unique_contexts = sorted(all_contexts_set)

    logger.info(
        "Unique texts: %d questions, %d answers, %d contexts",
        len(unique_questions), len(unique_answers), len(unique_contexts),
    )

    # Load compatible shared embeddings or explicitly refresh their caches.
    q_embs_unique = _load_or_compute_embeddings(
        unique_questions, "questions", shared_cache, encoder, expected_meta,
        force_recompute=force_recompute,
    )
    a_embs_unique = _load_or_compute_embeddings(
        unique_answers, "answers", shared_cache, encoder, expected_meta,
        force_recompute=force_recompute,
    )
    c_embs_unique = _load_or_compute_embeddings(
        unique_contexts, "contexts", shared_cache, encoder, expected_meta,
        force_recompute=force_recompute,
    )

    q_emb_map = {q: q_embs_unique[i] for i, q in enumerate(unique_questions)}
    a_emb_map = {a: a_embs_unique[i] for i, a in enumerate(unique_answers)}
    c_emb_map = {c: c_embs_unique[i] for i, c in enumerate(unique_contexts)}

    # Load compatible chunk embeddings or explicitly refresh their cache.
    all_chunk_embs = _load_or_compute_chunk_embeddings(
        unique_contexts, shared_cache, encoder, expected_meta,
        tokenizer, max_chunk_tokens, chunk_overlap, c_emb_map,
        force_recompute=force_recompute,
    )

    # Process each split
    results: Dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        logger.info("Processing split: %s", split)
        examples = split_data[split]
        n = len(examples)

        q_embs = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
        a_embs = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
        c_a_embs = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
        all_c_embs_list: List[np.ndarray] = []
        q_c_sim_maxes = np.zeros(n, dtype=np.float32)
        answer_texts: List[str] = []
        context_texts: List[str] = []

        for i, ex in enumerate(examples):
            q_emb = q_emb_map[ex["question"]]
            a_emb = a_emb_map[ex["candidate_answer"]]
            ctx_text = ex["reference_context"]
            chunk_embs = all_chunk_embs[ctx_text]

            sims = np.einsum("ij,j->i", chunk_embs, a_emb)
            best_idx = int(np.argmax(sims))
            c_a_emb = chunk_embs[best_idx]

            q_c_sims = np.einsum("ij,j->i", chunk_embs, q_emb)
            q_c_sim_max = float(np.max(q_c_sims))

            q_embs[i] = q_emb
            a_embs[i] = a_emb
            c_a_embs[i] = c_a_emb
            all_c_embs_list.append(chunk_embs)
            q_c_sim_maxes[i] = q_c_sim_max
            answer_texts.append(ex["candidate_answer"])
            context_texts.append(ctx_text)

        # Build features
        logger.info("Building features for %s (%d examples)...", split, n)
        features = build_features_batch(
            q_embs, a_embs, c_a_embs, all_c_embs_list,
            answer_texts, context_texts, q_c_sim_maxes,
        )

        labels = np.array([ex["label"] for ex in examples], dtype=np.int64)
        example_ids = [ex["example_id"] for ex in examples]
        source_ids = [ex["source_id"] for ex in examples]

        # Scale: fit on train only
        if split == "train":
            mean, scale = fit_scaler(features)
            save_scaler(SCALER_DIR / "scaler.npz", mean, scale)
            logger.info("Scaler fitted on training data (%d examples)", n)

        scaler_mean, scaler_scale = load_scaler(SCALER_DIR / "scaler.npz")
        scaled_features = transform_with_scaler(features, scaler_mean, scaler_scale)

        # Save artifacts
        output_dir = OUTPUT_DIR / split
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "features.npy", features)
        np.save(output_dir / "features_scaled.npy", scaled_features)
        np.save(output_dir / "labels.npy", labels)
        (output_dir / "example_ids.json").write_text(
            json.dumps(example_ids, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "source_ids.json").write_text(
            json.dumps(source_ids, ensure_ascii=False), encoding="utf-8"
        )

        meta = {
            "split": split,
            "row_count": n,
            "feature_dim": int(features.shape[1]),
            "label_counts": {
                "grounded": int((labels == 0).sum()),
                "hallucinated": int((labels == 1).sum()),
            },
            "has_nans": bool(np.any(np.isnan(features))),
            "has_infs": bool(np.any(np.isinf(features))),
            "scaled_has_nans": bool(np.any(np.isnan(scaled_features))),
            "scaled_has_infs": bool(np.any(np.isinf(scaled_features))),
            "feature_range": {
                "min": float(features.min()),
                "max": float(features.max()),
                "mean": float(features.mean()),
            },
            "scaler_mean_range": {
                "min": float(scaler_mean.min()),
                "max": float(scaler_mean.max()),
            },
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        results[split] = {
            "row_count": n,
            "feature_shape": list(features.shape),
            "scaled_feature_shape": list(scaled_features.shape),
            "label_counts": meta["label_counts"],
        }
        logger.info(
            "Split %s: features=%s, scaled=%s, labels=%s",
            split, features.shape, scaled_features.shape, meta["label_counts"],
        )

    elapsed = time.time() - t0
    results["timing_seconds"] = round(elapsed, 2)
    results["encoder"] = {
        "model": encoder.model_name,
        "revision": encoder.revision,
        "sentence_transformers_version": encoder.sentence_transformers_version,
    }
    results["unique_texts"] = {
        "questions": len(unique_questions),
        "answers": len(unique_answers),
        "contexts": len(unique_contexts),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Phase 3 pipeline completed in %.1f seconds", elapsed)
    return results
