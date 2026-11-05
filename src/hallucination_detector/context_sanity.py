"""Paired context derangement and Phase 3 feature recomputation from frozen caches."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from hallucination_detector.baselines.pipeline import _json
from hallucination_detector.data.duplicates import normalize_text as duplicate_normalize
from hallucination_detector.features.embeddings import _validate_cache_metadata
from hallucination_detector.features.preprocessing import normalize_text
from hallucination_detector.features.primary import build_features_batch


def context_mapping(rows: list[dict], seed: int = 42) -> tuple[dict[str, str], dict]:
    """Construct one seeded bijection excluding own and duplicate normalized contexts.

    Contiguous context groups rotated by the largest group size cannot overlap
    themselves when no group occupies more than half the sources. This avoids
    repeated random draws and fails explicitly when a valid assignment is impossible.
    """
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["source_id"]].append(row)
    groups = defaultdict(list)
    for sid, pair in pairs.items():
        if (len(pair) != 2 or sorted(r["label"] for r in pair) != [0, 1]
                or pair[0]["reference_context"] != pair[1]["reference_context"]):
            raise ValueError("Context shuffle requires intact balanced source pairs")
        groups[duplicate_normalize(pair[0]["reference_context"])].append(sid)
    if not groups:
        raise ValueError("Cannot shuffle empty sources")
    largest = max(map(len, groups.values()))
    if largest * 2 > len(pairs):
        raise ValueError("No valid context derangement: one normalized context exceeds half the sources")
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    ordered = []
    for index in rng.permutation(len(keys)):
        members = sorted(groups[keys[index]])
        ordered.extend(members[i] for i in rng.permutation(len(members)))
    mapping = {sid: ordered[(i + largest) % len(ordered)] for i, sid in enumerate(ordered)}
    contexts = {sid: duplicate_normalize(pair[0]["reference_context"]) for sid, pair in pairs.items()}
    checks = {
        "one_to_one": set(mapping) == set(mapping.values()) == set(pairs),
        "no_source_retains_own_context": all(s != d for s, d in mapping.items()),
        "no_identical_normalized_context": all(contexts[s] != contexts[d] for s, d in mapping.items()),
        "paired_sources": all(len(p) == 2 and sorted(r["label"] for r in p) == [0, 1] for p in pairs.values()),
    }
    if not all(checks.values()):
        raise ValueError("Context shuffle integrity failure")
    return mapping, {"seed": seed, "source_count": len(pairs), "example_count": len(rows),
                     "normalized_context_count": len(groups), "largest_context_group": largest,
                     "checks": checks, "all_passed": True,
                     "algorithm": "NumPy default_rng; sorted context groups and members, seeded group/member permutations, circular shift by largest group size"}


def apply_context_mapping(rows: list[dict], mapping: dict[str, str]) -> list[dict]:
    """Replace only reference contexts, retaining row order, questions, answers and IDs."""
    contexts = {r["source_id"]: r["reference_context"] for r in rows}
    if set(mapping) != set(contexts) or set(mapping.values()) != set(contexts):
        raise ValueError("Context mapping must be a bijection of the source IDs")
    if any(s == d or duplicate_normalize(contexts[s]) == duplicate_normalize(contexts[d]) for s, d in mapping.items()):
        raise ValueError("Own or identical normalized context replacement")
    return [{**row, "reference_context": contexts[mapping[row["source_id"]]]} for row in rows]


def load_frozen_cache(root: Path, expected_metadata: dict) -> dict:
    """Read compatible frozen caches only; never instantiate or run an encoder."""
    path = root / "data/embedding_cache"
    result = {}
    for variant in ("questions", "answers", "chunks"):
        meta = _json(path / f"cache_metadata_{variant}.json")
        expected = expected_metadata[variant]
        _validate_cache_metadata(meta, expected, path)
        if meta != expected:
            raise ValueError("Frozen cache encoder metadata changed after evaluation lock")
        texts = _json(path / f"embedding_ids_{variant}.json")
        embeddings = np.load(path / f"embeddings_{variant}.npy", allow_pickle=False)
        if (len(texts) != len(set(texts)) or any(normalize_text(t) != t for t in texts)
                or embeddings.ndim != 2 or embeddings.shape[1] != 384
                or not np.isfinite(embeddings).all()
                or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4)):
            raise ValueError("Invalid frozen embedding cache values/identifiers")
        if variant == "chunks":
            offsets = np.load(path / "chunk_offsets.npy", allow_pickle=False)
            if (offsets.shape != (len(texts) + 1,) or not np.issubdtype(offsets.dtype, np.integer)
                    or offsets[0] != 0 or offsets[-1] != len(embeddings) or (np.diff(offsets) <= 0).any()):
                raise ValueError("Invalid cached chunk offsets")
            result[variant] = {text: embeddings[offsets[i]:offsets[i+1]] for i, text in enumerate(texts)}
        else:
            if embeddings.shape[0] != len(texts):
                raise ValueError("Frozen cache identifier/embedding alignment mismatch")
            result[variant] = dict(zip(texts, embeddings))
    return result


def recompute_context_features(rows: list[dict], cache: dict) -> np.ndarray:
    """Reuse Phase 3 chunk selection and complete primary feature construction."""
    questions, answers, best, chunks, answer_texts, context_texts, qc = [], [], [], [], [], [], []
    for row in rows:
        q, a, c = (normalize_text(row[k]) for k in ("question", "candidate_answer", "reference_context"))
        try:
            q_emb, a_emb, chunk_embs = cache["questions"][q], cache["answers"][a], cache["chunks"][c]
        except KeyError as exc:
            raise ValueError("Evaluation input absent from compatible frozen cache") from exc
        sims = np.einsum("ij,j->i", chunk_embs, a_emb)
        questions.append(q_emb)
        answers.append(a_emb)
        best.append(chunk_embs[int(np.argmax(sims))])
        chunks.append(chunk_embs)
        qc.append(np.max(np.einsum("ij,j->i", chunk_embs, q_emb)))
        answer_texts.append(a)
        context_texts.append(c)
    return build_features_batch(np.asarray(questions), np.asarray(answers), np.asarray(best), chunks,
                                answer_texts, context_texts, np.asarray(qc, dtype=np.float32))
