"""Run Phase 2 leakage validation checks."""
from __future__ import annotations

import json
from pathlib import Path

from hallucination_detector.data.split_phase2 import (
    load_json_or_ndjson,
    load_phase1_validated_records,
    split_config_from_default,
    validate_phase2_leakage,
)


RAW_PATH = Path("data/raw/qa_data.json")
PHASE1_MANIFEST = Path("data/metadata/halueval_manifest.json")
PHASE1_VALIDATION_REPORT = Path("data/metadata/halueval_validation_report.json")
SPLIT_MANIFEST_PATH = Path("data/splits/halueval_original_split_manifest.json")
EXPANDED_ALL_PATH = Path("data/processed/halueval_expanded_all.jsonl")
OUT_REPORT = Path("data/splits/halueval_split_leakage_report.json")


def main() -> None:
    phase1_manifest = json.loads(PHASE1_MANIFEST.read_text(encoding="utf-8"))
    phase1_report = json.loads(PHASE1_VALIDATION_REPORT.read_text(encoding="utf-8"))
    split_manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    expanded_rows = load_json_or_ndjson(EXPANDED_ALL_PATH)
    records = load_phase1_validated_records(RAW_PATH, PHASE1_MANIFEST, PHASE1_VALIDATION_REPORT)
    proportions, seed = split_config_from_default()

    report = validate_phase2_leakage(
        phase1_manifest=phase1_manifest,
        phase1_exclusions=phase1_report.get("exclusions", []),
        split_manifest=split_manifest,
        expanded_rows=expanded_rows,
        split_seed=seed,
        split_proportions=proportions,
        records=records,
    )
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Leakage report written to {OUT_REPORT}")
    print(f"All checks passed: {report['all_passed']}")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

