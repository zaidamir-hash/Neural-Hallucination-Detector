"""Phase 3 entry point script.

Run the frozen-embedding feature pipeline:
  1. Load Phase 2 processed data
  2. Load frozen encoder
  3. Generate/cache embeddings
  4. Build 1,543-dim primary features
  5. Fit scaler on training data only
  6. Transform all splits
  7. Save artifacts and metadata

Usage:
    python scripts/run_phase3.py
    python scripts/run_phase3.py --force-recompute
"""
from __future__ import annotations

import json
import argparse
import logging
import sys
from pathlib import Path

# Ensure src is on path for editable install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallucination_detector.features.pipeline import run_phase3_pipeline
from hallucination_detector.features.encoder import FrozenEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_phase3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen-embedding Phase 3 features")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Explicitly replace all embedding caches before rebuilding features",
    )
    args = parser.parse_args()

    logger.info("Starting Phase 3 pipeline")
    logger.info("Loading frozen encoder...")
    encoder = FrozenEncoder()

    results = run_phase3_pipeline(
        encoder=encoder,
        force_recompute=args.force_recompute,
    )

    print("\n" + "=" * 60)
    print("Phase 3 Pipeline Results")
    print("=" * 60)
    for split in ("train", "validation", "test"):
        r = results[split]
        print(f"\n{split.upper()}:")
        print(f"  Row count:      {r['row_count']}")
        print(f"  Feature shape:  {r['feature_shape']}")
        print(f"  Labels:         {r['label_counts']}")
    print(f"\nTiming: {results['timing_seconds']:.1f}s")
    print(f"Encoder: {results['encoder']}")
    print(f"Unique texts: {results['unique_texts']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
