"""Configuration helpers for the experiment.

This module exposes a small loader that reads the canonical YAML configuration file
created in configs/default.yaml. PyYAML is optional at runtime; if not installed,
users are instructed to install it to access the parsed config via Python API.
"""
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2].joinpath("configs", "default.yaml")


def load_config() -> Dict[str, Any]:
    """Load the canonical YAML configuration.

    Returns a parsed dict if PyYAML is available. If PyYAML is not installed,
    raises an informative RuntimeError telling the user how to install it.
    """
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required to parse the YAML configuration. Install with: pip install pyyaml"
        ) from exc

    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


if __name__ == "__main__":
    # Quick CLI check
    try:
        cfg = load_config()
        print("Loaded config keys:", list(cfg.keys()))
    except RuntimeError as e:
        print(e)
