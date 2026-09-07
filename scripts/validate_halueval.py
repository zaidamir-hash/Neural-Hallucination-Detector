"""Script to validate HaluEval QA raw records and assign source_ids."""
from pathlib import Path
from hallucination_detector.data.validate import validate_records
import json

RAW = Path("data/raw/qa_data.json")
OUT = Path("data/metadata/halueval_validation_report.json")


def main():
    annotated, counts, exclusions = validate_records(RAW)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {"counts": counts, "exclusions": exclusions}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # write annotated source ids as a manifest
    manifest = [ {"original_index": a["original_index"], "source_id": a["source_id"]} for a in annotated]
    Path("data/metadata/halueval_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Validation complete. Counts: {counts}. Report: {OUT}")

if __name__ == "__main__":
    main()
