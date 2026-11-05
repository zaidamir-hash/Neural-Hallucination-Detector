"""Fit baseline models using training/validation only; do not evaluate the test split."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.baselines.pipeline import run_phase4


if __name__ == "__main__":
    print(json.dumps(run_phase4(ROOT), indent=2))
