"""Phase 2 leakage-safe splitting and paired expansion utilities."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Optional, Tuple

from hallucination_detector.config import load_config
from hallucination_detector.data.acquire import utc_timestamp_z
from hallucination_detector.data.duplicates import normalize_text
from hallucination_detector.data.validate import REQUIRED_FIELDS


def load_json_or_ndjson(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        raise ValueError("Unsupported JSON top-level type")
    except json.JSONDecodeError:
        lines = [l for l in text.splitlines() if l.strip()]
        return [json.loads(l) for l in lines]


def load_phase1_validated_records(
    raw_path: Path,
    phase1_manifest_path: Path,
    phase1_validation_report_path: Path,
) -> List[Dict[str, Any]]:
    """Load validated original records from Phase 1 artifacts."""
    raw_records = load_json_or_ndjson(raw_path)
    manifest = json.loads(phase1_manifest_path.read_text(encoding="utf-8"))
    validation_report = json.loads(phase1_validation_report_path.read_text(encoding="utf-8"))
    exclusions = validation_report.get("exclusions", [])
    excluded_indices = {int(item[0]) for item in exclusions}

    index_to_source: Dict[int, str] = {}
    for item in manifest:
        idx = int(item["original_index"])
        sid = item["source_id"]
        if idx in index_to_source:
            raise ValueError(f"Duplicate original_index in Phase 1 manifest: {idx}")
        index_to_source[idx] = sid

    if len(index_to_source) != len(raw_records):
        raise ValueError(
            f"Phase 1 manifest size ({len(index_to_source)}) does not match raw record count ({len(raw_records)})"
        )

    valid_records: List[Dict[str, Any]] = []
    for idx, rec in enumerate(raw_records):
        if idx in excluded_indices:
            continue
        if idx not in index_to_source:
            raise ValueError(f"Missing source_id for index {idx} in Phase 1 manifest")
        if not isinstance(rec, dict):
            raise ValueError(f"Invalid record at index {idx}: expected object")
        for field in REQUIRED_FIELDS:
            if field not in rec:
                raise ValueError(f"Missing required field '{field}' at index {idx}")
        valid_records.append(
            {
                "original_index": idx,
                "source_id": index_to_source[idx],
                "question": rec["question"],
                "knowledge": rec["knowledge"],
                "right_answer": rec["right_answer"],
                "hallucinated_answer": rec["hallucinated_answer"],
            }
        )
    return valid_records


def _context_group_id(context: str) -> str:
    norm = normalize_text(context)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def split_config_from_default() -> Tuple[Dict[str, float], int]:
    cfg = load_config()
    split_cfg = cfg["split"]
    proportions = {
        "train": float(split_cfg["train"]),
        "validation": float(split_cfg["validation"]),
        "test": float(split_cfg["test"]),
    }
    if abs(sum(proportions.values()) - 1.0) > 1e-9:
        raise ValueError(f"Split proportions must sum to 1.0, got {sum(proportions.values())}")
    seed = int(split_cfg["seed"])
    return proportions, seed


def assign_group_aware_splits(
    records: List[Dict[str, Any]],
    proportions: Mapping[str, float],
    seed: int,
) -> List[Dict[str, Any]]:
    """Assign split labels at original-record level with duplicate-context grouping."""
    split_names = ["train", "validation", "test"]
    for s in split_names:
        if s not in proportions:
            raise ValueError(f"Missing split proportion for '{s}'")
    n_total = len(records)
    targets = {
        "train": int(round(n_total * proportions["train"])),
        "validation": int(round(n_total * proportions["validation"])),
    }
    targets["test"] = n_total - targets["train"] - targets["validation"]

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        gid = _context_group_id(rec["knowledge"])
        groups[gid].append(i)

    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    counts = {k: 0 for k in split_names}
    group_to_split: Dict[str, str] = {}
    split_order = ["train", "validation", "test"]

    for gid in group_ids:
        group_size = len(groups[gid])
        deficits = {s: targets[s] - counts[s] for s in split_names}
        positive_deficits = [s for s in split_names if deficits[s] > 0]
        if positive_deficits:
            # Prefer split with largest remaining deficit; tie-break by fixed split order.
            chosen = sorted(
                positive_deficits,
                key=lambda s: (-deficits[s], split_order.index(s)),
            )[0]
        else:
            # If all targets are met/exceeded, minimize overflow impact.
            chosen = sorted(
                split_names,
                key=lambda s: (
                    abs((counts[s] + group_size) - targets[s]),
                    split_order.index(s),
                ),
            )[0]
        group_to_split[gid] = chosen
        counts[chosen] += group_size

    manifest: List[Dict[str, Any]] = []
    seen_source_ids = set()
    for rec in records:
        gid = _context_group_id(rec["knowledge"])
        split = group_to_split[gid]
        source_id = rec["source_id"]
        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate source_id in validated records: {source_id}")
        seen_source_ids.add(source_id)
        manifest.append(
            {
                "source_id": source_id,
                "original_index": rec["original_index"],
                "duplicate_context_group_id": gid,
                "split": split,
            }
        )

    return manifest


def build_split_metadata(
    manifest: List[Dict[str, Any]],
    dataset_checksum: str,
    source_revision: str,
    split_seed: int,
    requested_proportions: Mapping[str, float],
) -> Dict[str, Any]:
    total = len(manifest)
    counts = {"train": 0, "validation": 0, "test": 0}
    for row in manifest:
        counts[row["split"]] += 1
    achieved = {k: (counts[k] / total if total else 0.0) for k in counts}
    return {
        "dataset_checksum": dataset_checksum,
        "source_revision": source_revision,
        "split_seed": split_seed,
        "requested_proportions": dict(requested_proportions),
        "achieved_counts": counts,
        "achieved_proportions": achieved,
        "grouping_rule": "Group by SHA-256(normalize_text(knowledge)); assign groups to splits deterministically with seed.",
        "creation_timestamp_utc": utc_timestamp_z(),
    }


def deterministic_example_id(source_id: str, answer_variant: str) -> str:
    raw = f"{source_id}|{answer_variant}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _exact_record_group_id(record: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {
            "question": normalize_text(record.get("question", "")),
            "knowledge": normalize_text(record.get("knowledge", "")),
            "right_answer": normalize_text(record.get("right_answer", "")),
            "hallucinated_answer": normalize_text(record.get("hallucinated_answer", "")),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def expand_records_to_examples(
    records: List[Dict[str, Any]],
    manifest: List[Dict[str, Any]],
    labels: Mapping[str, int],
) -> List[Dict[str, Any]]:
    index_by_source = {r["source_id"]: r for r in records}
    expanded: List[Dict[str, Any]] = []

    grounded_label = int(labels["grounded"])
    hallucinated_label = int(labels["hallucinated"])
    if grounded_label != 0 or hallucinated_label != 1:
        raise ValueError("Label convention must remain 0=Grounded and 1=Hallucinated")

    for row in manifest:
        source_id = row["source_id"]
        split = row["split"]
        rec = index_by_source.get(source_id)
        if rec is None:
            raise ValueError(f"Unknown source_id in manifest: {source_id}")
        expanded.append(
            {
                "example_id": deterministic_example_id(source_id, "right_answer"),
                "source_id": source_id,
                "split": split,
                "question": rec["question"],
                "reference_context": rec["knowledge"],
                "candidate_answer": rec["right_answer"],
                "label": grounded_label,
                "label_name": "Grounded",
                "answer_variant": "right_answer",
            }
        )
        expanded.append(
            {
                "example_id": deterministic_example_id(source_id, "hallucinated_answer"),
                "source_id": source_id,
                "split": split,
                "question": rec["question"],
                "reference_context": rec["knowledge"],
                "candidate_answer": rec["hallucinated_answer"],
                "label": hallucinated_label,
                "label_name": "Hallucinated",
                "answer_variant": "hallucinated_answer",
            }
        )
    return expanded


def validate_phase2_leakage(
    phase1_manifest: List[Dict[str, Any]],
    phase1_exclusions: List[List[Any]],
    split_manifest: List[Dict[str, Any]],
    expanded_rows: List[Dict[str, Any]],
    split_seed: int,
    split_proportions: Mapping[str, float],
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run strict leakage and consistency checks required for Phase 2."""
    checks: Dict[str, bool] = {}

    # Valid source IDs from Phase 1
    excluded_indices = {int(e[0]) for e in phase1_exclusions}
    valid_phase1_source_ids = {
        item["source_id"]
        for item in phase1_manifest
        if int(item["original_index"]) not in excluded_indices
    }
    split_source_ids = [row["source_id"] for row in split_manifest]

    checks["every_valid_phase1_source_id_appears_once"] = (
        len(split_source_ids) == len(set(split_source_ids)) == len(valid_phase1_source_ids)
        and set(split_source_ids) == valid_phase1_source_ids
    )
    checks["no_unknown_or_missing_source_ids"] = set(split_source_ids) == valid_phase1_source_ids

    # No source ID crosses splits
    source_to_split: Dict[str, str] = {}
    for row in split_manifest:
        sid = row["source_id"]
        split = row["split"]
        if sid in source_to_split and source_to_split[sid] != split:
            checks["no_source_id_crosses_splits"] = False
            break
        source_to_split[sid] = split
    else:
        checks["no_source_id_crosses_splits"] = True

    # No duplicate-context group crosses splits
    group_to_split: Dict[str, str] = {}
    for row in split_manifest:
        gid = row["duplicate_context_group_id"]
        split = row["split"]
        if gid in group_to_split and group_to_split[gid] != split:
            checks["no_duplicate_context_group_crosses_splits"] = False
            break
        group_to_split[gid] = split
    else:
        checks["no_duplicate_context_group_crosses_splits"] = True

        # No known exact duplicate group crosses splits
        source_to_record = {r["source_id"]: r for r in records}
        exact_group_to_split: Dict[str, str] = {}
        for row in split_manifest:
            sid = row["source_id"]
            split = row["split"]
            rec = source_to_record[sid]
            egid = _exact_record_group_id(rec)
            if egid in exact_group_to_split and exact_group_to_split[egid] != split:
                checks["no_exact_duplicate_group_crosses_splits"] = False
                break
            exact_group_to_split[egid] = split
        else:
            checks["no_exact_duplicate_group_crosses_splits"] = True

    # Expanded checks
    source_expanded_counts = defaultdict(int)
    source_expanded_split = defaultdict(set)
    label_counts_by_split = {"train": defaultdict(int), "validation": defaultdict(int), "test": defaultdict(int)}
    example_ids = set()
    deterministic_ok = True
    labels_ok = True
    for row in expanded_rows:
        sid = row["source_id"]
        source_expanded_counts[sid] += 1
        source_expanded_split[sid].add(row["split"])
        label_counts_by_split[row["split"]][row["label"]] += 1
        expected = deterministic_example_id(sid, row["answer_variant"])
        if row["example_id"] != expected:
            deterministic_ok = False
        if row["label"] == 0 and row["label_name"] != "Grounded":
            labels_ok = False
        if row["label"] == 1 and row["label_name"] != "Hallucinated":
            labels_ok = False
        if row["label"] not in (0, 1):
            labels_ok = False
        example_ids.add(row["example_id"])

    checks["both_examples_from_source_in_same_split"] = all(len(splits) == 1 for splits in source_expanded_split.values())
    checks["every_source_produces_exactly_two_examples"] = all(v == 2 for v in source_expanded_counts.values()) and len(source_expanded_counts) == len(split_manifest)
    checks["every_split_balanced_grounded_hallucinated"] = all(
        label_counts_by_split[s][0] == label_counts_by_split[s][1] for s in label_counts_by_split
    )
    checks["labels_are_0_grounded_1_hallucinated"] = labels_ok
    checks["example_ids_unique_and_deterministic"] = deterministic_ok and len(example_ids) == len(expanded_rows)

    # Reproducibility with same seed
    repeat_manifest = assign_group_aware_splits(records, split_proportions, split_seed)
    checks["same_seed_reproduces_manifest"] = split_manifest == repeat_manifest

    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "split_manifest_count": len(split_manifest),
        "expanded_row_count": len(expanded_rows),
    }
