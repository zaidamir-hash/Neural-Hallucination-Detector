"""Validation and source_id assignment for HaluEval QA original records."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import hashlib
import re

REQUIRED_FIELDS = ["question", "knowledge", "right_answer", "hallucinated_answer"]


def canonical_record_str(record: Dict[str, Any]) -> str:
    """Create a canonical JSON string for hashing from required fields.

    Uses minimal normalization: convert to strings, normalize whitespace, and order keys.
    """
    def norm(s: Any) -> str:
        if s is None:
            return ""
        s = str(s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    obj = {k: norm(record.get(k, "")) for k in sorted(REQUIRED_FIELDS)}
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def compute_source_id(record: Dict[str, Any]) -> str:
    s = canonical_record_str(record).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def validate_records(raw_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[Tuple[int, str]]]:
    """Validate original records in the raw JSON file.

    Returns:
      - list of records annotated with source_id and original_index
      - counts dict {total, valid, excluded}
      - list of exclusions as tuples (index, reason)
    """
    text = raw_path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
        if isinstance(raw, dict):
            raw = [raw]
    except Exception:
        # NDJSON support
        lines = [l for l in text.splitlines() if l.strip()]
        raw = [json.loads(l) for l in lines]

    total = len(raw)
    annotated = []
    exclusions: List[Tuple[int, str]] = []

    for idx, rec in enumerate(raw):
        reason = None
        if not isinstance(rec, dict):
            reason = "non_object"
            exclusions.append((idx, reason))
            # still compute source id with empty canonicalization
            source_id = hashlib.sha256(f"{idx}".encode("utf-8")).hexdigest()
            annotated.append({"original_index": idx, "source_id": source_id, "record": rec})
            continue

        # check required fields presence
        for f in REQUIRED_FIELDS:
            if f not in rec:
                reason = f"missing_field:{f}"
                break
            if rec[f] is None:
                reason = f"null_field:{f}"
                break
            if not isinstance(rec[f], str):
                reason = f"non_string_field:{f}"
                break
            if rec[f].strip() == "":
                reason = f"empty_field:{f}"
                break

        # identical answers check
        if reason is None:
            ra = re.sub(r"\s+", " ", rec.get("right_answer", "").strip())
            ha = re.sub(r"\s+", " ", rec.get("hallucinated_answer", "").strip())
            if ra == ha:
                reason = "identical_answers"

        source_id = compute_source_id(rec)
        annotated.append({"original_index": idx, "source_id": source_id, "record": rec})
        if reason is not None:
            exclusions.append((idx, reason))

    valid = total - len(exclusions)
    counts = {"total": total, "valid": valid, "excluded": len(exclusions)}
    return annotated, counts, exclusions


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=str, default="data/raw/qa_data.json")
    args = p.parse_args()

    raw_path = Path(args.raw)
    annotated, counts, exclusions = validate_records(raw_path)
    print("Total:", counts["total"]) 
    print("Valid:", counts["valid"]) 
    print("Excluded:", counts["excluded"]) 
    if exclusions:
        print("Sample exclusions:", exclusions[:10])
