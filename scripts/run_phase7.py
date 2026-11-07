"""Export qualitative reviews from saved representative MLP predictions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.error_analysis import OUTPUT, export_phase7
from hallucination_detector.reproduction import REFERENCE, WORKSPACE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    notes_dir = REFERENCE if (ROOT / WORKSPACE).is_file() else OUTPUT
    parser.add_argument("--notes", type=Path, default=ROOT / notes_dir / "analyst_notes.json")
    args = parser.parse_args()
    manifest = export_phase7(ROOT, args.notes)
    keys = ("model", "seed", "threshold", "available", "reviewed", "review_candidates", "review_status")
    print(json.dumps({k: manifest[k] for k in keys if k in manifest}, indent=2))
