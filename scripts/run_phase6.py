"""Prepare the pre-test lock/smoke check, or perform the single locked evaluation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.evaluation import prepare_evaluation
from hallucination_detector.evaluation_pipeline import run_final_evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="Lock saved decisions and verify validation inference; no test access")
    mode.add_argument("--evaluate", action="store_true", help="Run once after synthetic tests and successful preparation")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare_evaluation(ROOT), indent=2))
    else:
        report = run_final_evaluation(ROOT)
        print(json.dumps({"results": report["primary"]["results"],
                          "mlp_summary": report["primary"]["mlp_three_seed_summary"],
                          "shuffled": report["shuffled"]}, indent=2))
