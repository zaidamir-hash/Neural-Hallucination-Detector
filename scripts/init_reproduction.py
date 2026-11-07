"""Initialize an independent experiment in a new directory."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.reproduction import initialize_workspace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    try:
        initialize_workspace(ROOT, args.target)
    except FileExistsError:
        parser.error("Target already exists. Choose a new directory.")
