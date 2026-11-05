"""Generate the descriptive Phase 3 semantic audit using training data only."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.features.semantic_audit import run_train_semantic_audit


if __name__ == "__main__":
    print(json.dumps(run_train_semantic_audit(ROOT), indent=2))
