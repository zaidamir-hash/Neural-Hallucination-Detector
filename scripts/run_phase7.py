"""Export qualitative reviews from saved original MLP seed 202 predictions only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.error_analysis import OUTPUT, export_phase7


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, default=ROOT / OUTPUT / "analyst_notes.json")
    args = parser.parse_args()
    manifest = export_phase7(ROOT, args.notes)
    print(json.dumps({k: manifest[k] for k in ("model", "seed", "threshold", "available", "reviewed")}, indent=2))
