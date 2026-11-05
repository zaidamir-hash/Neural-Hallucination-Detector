"""Primary feature generation.

Constructs the locked 1,543-dimensional primary feature vector from
frozen embeddings and lexical/numeric statistics.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from hallucination_detector.features.encoder import EMBEDDING_DIM
from hallucination_detector.features.scalar_features import (
    compute_answer_content_coverage,
    compute_answer_numeric_coverage,
    compute_answer_has_numeric,
)


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. Assumes L2-normalized inputs."""
    return np.einsum("ij,ij->i", a, b)


def build_primary_features(
    q_emb: np.ndarray,
    a_emb: np.ndarray,
    c_a_emb: np.ndarray,
    all_c_embs: np.ndarray,
    answer_text: str,
    context_text: str,
    q_c_sim_max: float,
) -> np.ndarray:
    """Build the 1,543-dim primary feature vector.

    Args:
        q_emb: (384,) question embedding
        a_emb: (384,) answer embedding
        c_a_emb: (384,) best context chunk embedding
        all_c_embs: (n_chunks, 384) all context chunk embeddings
        answer_text: raw answer text for lexical/numeric features
        context_text: raw context text for lexical/numeric features
        q_c_sim_max: precomputed max question-context cosine similarity

    Returns:
        (1543,) float32 feature vector
    """
    # Ensure 1-D
    q_emb = q_emb.ravel()
    a_emb = a_emb.ravel()
    c_a_emb = c_a_emb.ravel()

    # -- Four 384-dim interaction blocks --
    block1 = np.abs(a_emb - c_a_emb)          # |a - c_a|
    block2 = a_emb * c_a_emb                   # a ⊙ c_a
    block3 = np.abs(q_emb - a_emb)             # |q - a|
    block4 = q_emb * a_emb                     # q ⊙ a

    vector_blocks = np.concatenate([block1, block2, block3, block4])  # (1536,)

    # -- Seven scalar features --
    # 1. Max answer-context cosine
    c_sims = _cosine_sim_matrix(
        a_emb.reshape(1, -1),
        all_c_embs,
    )
    max_ac_sim = float(c_sims.max())

    # 2. Mean top-3 answer-context cosine
    sorted_sims = np.sort(c_sims)[::-1]
    if len(sorted_sims) >= 3:
        mean_top3 = float(sorted_sims[:3].mean())
    elif len(sorted_sims) > 0:
        mean_top3 = float(sorted_sims.mean())
    else:
        mean_top3 = 0.0

    # 3. Question-answer cosine
    qa_sim = float(np.dot(q_emb, a_emb))

    # 4. Max question-context cosine (precomputed)
    max_qc_sim = float(q_c_sim_max)

    # 5. Answer content-token coverage in context
    content_cov = compute_answer_content_coverage(answer_text, context_text)

    # 6. Answer numeric-token coverage in context
    num_cov = compute_answer_numeric_coverage(answer_text, context_text)

    # 7. Numeric-token presence indicator
    has_num = compute_answer_has_numeric(answer_text)

    scalars = np.array(
        [max_ac_sim, mean_top3, qa_sim, max_qc_sim, content_cov, num_cov, has_num],
        dtype=np.float32,
    )

    features = np.concatenate([vector_blocks, scalars])  # (1543,)
    assert features.shape == (1543,), f"Feature shape mismatch: {features.shape}"
    return features


def build_features_batch(
    q_embs: np.ndarray,
    a_embs: np.ndarray,
    c_a_embs: np.ndarray,
    all_c_embs_list: List[np.ndarray],
    answer_texts: List[str],
    context_texts: List[str],
    q_c_sim_maxes: np.ndarray,
) -> np.ndarray:
    """Build features for a batch of examples.

    Args:
        q_embs: (N, 384)
        a_embs: (N, 384)
        c_a_embs: (N, 384)
        all_c_embs_list: list of (n_i, 384) arrays
        answer_texts: N answer strings
        context_texts: N context strings
        q_c_sim_maxes: (N,) max question-context similarities

    Returns:
        (N, 1543) float32 feature matrix
    """
    n = len(q_embs)
    features = np.zeros((n, 1543), dtype=np.float32)
    for i in range(n):
        features[i] = build_primary_features(
            q_embs[i],
            a_embs[i],
            c_a_embs[i],
            all_c_embs_list[i],
            answer_texts[i],
            context_texts[i],
            float(q_c_sim_maxes[i]),
        )
    return features
