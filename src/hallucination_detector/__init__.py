"""Neural Hallucination Detector package (Phase 0)

This package root contains minimal runtime helpers and exposes a config loader.
"""

__version__ = "0.0.0-dev"

from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2].joinpath("configs", "default.yaml")

__all__ = ["DEFAULT_CONFIG_PATH"]
