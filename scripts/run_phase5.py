"""Train the three locked MLP seeds using saved development features only."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.mlp_pipeline import run_phase5


if __name__ == "__main__":
    report = run_phase5(ROOT)
    print(json.dumps({"runs": report["runs"], "summary": report["summary"]}, indent=2))
