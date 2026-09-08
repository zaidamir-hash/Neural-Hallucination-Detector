"""Expand split original records into paired labeled examples (Phase 2)."""
from __future__ import annotations

import json
from pathlib import Path

from hallucination_detector.config import load_config
from hallucination_detector.data.split_phase2 import (
    expand_records_to_examples,
    load_phase1_validated_records,
)


RAW_PATH = Path("data/raw/qa_data.json")
PHASE1_MANIFEST = Path("data/metadata/halueval_manifest.json")
PHASE1_VALIDATION_REPORT = Path("data/metadata/halueval_validation_report.json")
SPLIT_MANIFEST_PATH = Path("data/splits/halueval_original_split_manifest.json")
OUT_ALL = Path("data/processed/halueval_expanded_all.jsonl")
OUT_BY_SPLIT = {
    "train": Path("data/processed/halueval_expanded_train.jsonl"),
    "validation": Path("data/processed/halueval_expanded_validation.jsonl"),
    "test": Path("data/processed/halueval_expanded_test.jsonl"),
}
OUT_COUNTS = Path("data/splits/halueval_expanded_counts.json")


def _write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = load_config()
    labels = cfg["labels"]
    records = load_phase1_validated_records(RAW_PATH, PHASE1_MANIFEST, PHASE1_VALIDATION_REPORT)
    manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    expanded = expand_records_to_examples(records, manifest, labels)

    _write_jsonl(expanded, OUT_ALL)
    split_rows = {k: [] for k in OUT_BY_SPLIT}
    for row in expanded:
        split_rows[row["split"]].append(row)
    for split_name, path in OUT_BY_SPLIT.items():
        _write_jsonl(split_rows[split_name], path)

    counts = {
        "total_expanded_rows": len(expanded),
        "by_split": {k: len(v) for k, v in split_rows.items()},
        "by_split_and_label": {
            k: {
                "grounded_label_0": sum(1 for r in v if r["label"] == 0),
                "hallucinated_label_1": sum(1 for r in v if r["label"] == 1),
            }
            for k, v in split_rows.items()
        },
    }
    OUT_COUNTS.write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Expanded rows written to {OUT_ALL}")
    for split_name, path in OUT_BY_SPLIT.items():
        print(f"{split_name}: {path} ({len(split_rows[split_name])} rows)")
    print(f"Expanded counts written to {OUT_COUNTS}")


if __name__ == "__main__":
    main()

