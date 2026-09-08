"""Create Phase 2 split manifest and split metadata."""
from __future__ import annotations

import json
from pathlib import Path

from hallucination_detector.data.split_phase2 import (
    assign_group_aware_splits,
    build_split_metadata,
    load_phase1_validated_records,
    split_config_from_default,
)


RAW_PATH = Path("data/raw/qa_data.json")
PHASE1_MANIFEST = Path("data/metadata/halueval_manifest.json")
PHASE1_VALIDATION_REPORT = Path("data/metadata/halueval_validation_report.json")
PROVENANCE_PATH = Path("data/metadata/halueval_qa_provenance.json")
SPLIT_MANIFEST_PATH = Path("data/splits/halueval_original_split_manifest.json")
SPLIT_METADATA_PATH = Path("data/splits/halueval_split_metadata.json")


def main() -> None:
    records = load_phase1_validated_records(RAW_PATH, PHASE1_MANIFEST, PHASE1_VALIDATION_REPORT)
    proportions, seed = split_config_from_default()
    manifest = assign_group_aware_splits(records, proportions, seed)

    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    metadata = build_split_metadata(
        manifest=manifest,
        dataset_checksum=provenance["sha256"],
        source_revision=provenance["source_revision"],
        split_seed=seed,
        requested_proportions=proportions,
    )

    SPLIT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    SPLIT_METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = metadata["achieved_counts"]
    achieved = metadata["achieved_proportions"]
    print(f"Split manifest written to {SPLIT_MANIFEST_PATH}")
    print(f"Split metadata written to {SPLIT_METADATA_PATH}")
    print(
        "Counts: "
        f"train={counts['train']} ({achieved['train']:.4f}), "
        f"validation={counts['validation']} ({achieved['validation']:.4f}), "
        f"test={counts['test']} ({achieved['test']:.4f})"
    )


if __name__ == "__main__":
    main()

