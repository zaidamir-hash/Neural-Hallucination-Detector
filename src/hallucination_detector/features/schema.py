"""Load the feature schema from the canonical JSON file."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "configs" / "feature_schema.json"


def load_feature_schema() -> Dict[str, Any]:
    """Load and return the feature schema dict."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def get_feature_dim() -> int:
    """Return the total primary feature dimension (1543)."""
    schema = load_feature_schema()
    return schema["total_dim"]


def get_scalar_feature_names() -> list:
    """Return the ordered list of scalar feature names."""
    schema = load_feature_schema()
    return [f["name"] for f in schema["scalar_features"]["order"]]
