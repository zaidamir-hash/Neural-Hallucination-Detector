"""Duplicate auditing utilities for Phase 1."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional


def normalize_text(s: Any) -> str:
    """Normalize text for duplicate checks: lowercase, collapse whitespace, strip."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _exact_record_key(rec: Dict[str, Any]) -> str:
    return json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _extract_duplicate_groups(
    grouped: Mapping[str, List[int]],
    source_ids_by_index: Optional[Mapping[int, str]],
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for norm_key, indices in grouped.items():
        if len(indices) < 2:
            continue
        item: Dict[str, Any] = {
            "group_key_hash": _hash_text(norm_key),
            "group_size": len(indices),
            "original_indices": sorted(indices),
        }
        if source_ids_by_index is not None:
            item["source_ids"] = [source_ids_by_index.get(i) for i in sorted(indices)]
        groups.append(item)
    groups.sort(key=lambda g: (-g["group_size"], g["group_key_hash"]))
    return groups


def _summarize_groups(groups: List[Dict[str, Any]]) -> Dict[str, int]:
    involved = set()
    largest = 0
    for g in groups:
        involved.update(g["original_indices"])
        largest = max(largest, g["group_size"])
    return {
        "duplicate_group_count": len(groups),
        "unique_records_involved": len(involved),
        "largest_group_size": largest,
    }


def _groups_index_set(groups: Iterable[Dict[str, Any]]) -> set[int]:
    out: set[int] = set()
    for g in groups:
        out.update(g["original_indices"])
    return out


def audit_duplicates(
    records: List[Dict[str, Any]],
    source_ids_by_index: Optional[Mapping[int, str]] = None,
) -> Dict[str, Any]:
    """Build compact duplicate-audit artifact with hashed keys and exact summaries."""
    exact_map: Dict[str, List[int]] = defaultdict(list)
    q_map: Dict[str, List[int]] = defaultdict(list)
    k_map: Dict[str, List[int]] = defaultdict(list)
    qc_map: Dict[str, List[int]] = defaultdict(list)
    ra_map: Dict[str, List[int]] = defaultdict(list)
    ha_map: Dict[str, List[int]] = defaultdict(list)

    for idx, rec in enumerate(records):
        rec = rec if isinstance(rec, dict) else {}
        q = normalize_text(rec.get("question", ""))
        k = normalize_text(rec.get("knowledge", ""))
        ra = normalize_text(rec.get("right_answer", ""))
        ha = normalize_text(rec.get("hallucinated_answer", ""))

        exact_map[_exact_record_key(rec)].append(idx)
        q_map[q].append(idx)
        k_map[k].append(idx)
        qc_map[f"{q}\n{k}"].append(idx)
        ra_map[ra].append(idx)
        ha_map[ha].append(idx)

    categories = {
        "exact_complete_records": _extract_duplicate_groups(exact_map, source_ids_by_index),
        "normalized_questions": _extract_duplicate_groups(q_map, source_ids_by_index),
        "normalized_contexts": _extract_duplicate_groups(k_map, source_ids_by_index),
        "normalized_question_context_pairs": _extract_duplicate_groups(qc_map, source_ids_by_index),
        "grounded_answers": _extract_duplicate_groups(ra_map, source_ids_by_index),
        "hallucinated_answers": _extract_duplicate_groups(ha_map, source_ids_by_index),
    }

    summary = {name: _summarize_groups(groups) for name, groups in categories.items()}

    # Logical invariant: every duplicate QC group implies membership in duplicate-question and duplicate-context sets.
    qc_indices = _groups_index_set(categories["normalized_question_context_pairs"])
    q_indices = _groups_index_set(categories["normalized_questions"])
    k_indices = _groups_index_set(categories["normalized_contexts"])
    invariants = {
        "qc_implies_q_and_k_duplicate_membership": qc_indices.issubset(q_indices) and qc_indices.issubset(k_indices)
    }

    return {
        "normalization": {
            "function": "normalize_text",
            "description": "lowercase -> collapse whitespace -> strip",
        },
        "summary": summary,
        "invariants": invariants,
        "categories": categories,
    }


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/qa_data.json")
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except Exception:
        lines = [l for l in text.splitlines() if l.strip()]
        data = [json.loads(l) for l in lines]
    out = audit_duplicates(data)
    Path("results/metrics/halueval_duplicate_audit.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Duplicate audit written to results/metrics/halueval_duplicate_audit.json")
