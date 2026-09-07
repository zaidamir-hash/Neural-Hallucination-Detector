"""HaluEval QA acquisition utilities.

Provides a reproducible downloader for the official HaluEval QA dataset (qa_data.json)
from the RUCAIBox/HaluEval repository. The default is a revision-pinned raw URL.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import urllib.error

DEFAULT_COMMIT = "207f479c555a68b8d2431631a95083b84b2accbe"
DEFAULT_RAW_PATH = Path("data") / "raw" / "qa_data.json"
REQUIRED_FIELDS = ["question", "knowledge", "right_answer", "hallucinated_answer"]


def _default_raw_url(commit: str = DEFAULT_COMMIT) -> str:
    return f"https://raw.githubusercontent.com/RUCAIBox/HaluEval/{commit}/data/qa_data.json"


def compute_sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def utc_timestamp_z() -> str:
    """Return an explicit timezone-aware UTC timestamp in ISO-8601 with trailing Z."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def download_halueval_qa(
    out_path: Optional[Path] = None,
    url: Optional[str] = None,
    timeout: int = 30,
    force: bool = False,
) -> Tuple[Path, str, int]:
    """Download the official HaluEval QA JSON file.

    Args:
        out_path: output path under data/raw/ (default: data/raw/qa_data.json)
        url: explicit raw file URL. If omitted, uses the pinned commit DEFAULT_COMMIT.
        timeout: network timeout in seconds.
        force: if True, overwrite existing file.

    Returns:
        Tuple of (saved_path, sha256_hex, bytes_written)

    Raises:
        FileExistsError: if the destination exists and force is False.
        urllib.error.URLError: on network errors.
        ValueError: if the downloaded content is not valid JSON or required fields are absent.
    """
    if out_path is None:
        out_path = DEFAULT_RAW_PATH
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        raise FileExistsError(f"Raw file already exists at {out_path}. Use force=True to overwrite.")

    download_url = url or _default_raw_url()

    try:
        with urllib.request.urlopen(download_url, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        raise

    # Validate JSON: support both a JSON array and NDJSON (newline-delimited JSON)
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        # Try NDJSON: split lines and parse each line as JSON
        decoded = raw.decode("utf-8")
        lines = [l for l in decoded.splitlines() if l.strip()]
        try:
            parsed = [json.loads(l) for l in lines]
        except Exception as e:
            raise ValueError(f"Downloaded content is not valid JSON or NDJSON: {e}") from e

    if not isinstance(parsed, list):
        raise ValueError("Expected top-level JSON array or NDJSON lines of records in qa_data.json")

    if len(parsed) == 0:
        raise ValueError("Downloaded JSON contains zero records")

    # Quick schema check: ensure required fields are present in record objects
    missing_fields = set()
    for r in parsed:
        if not isinstance(r, dict):
            continue
        for f in REQUIRED_FIELDS:
            if f not in r:
                missing_fields.add(f)
    if missing_fields:
        raise ValueError(f"Required fields missing from dataset records: {sorted(list(missing_fields))}")

    sha256 = compute_sha256_bytes(raw)
    # write atomically
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        fh.write(raw)
    tmp.replace(out_path)
    size = out_path.stat().st_size

    return out_path, sha256, size


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Download HaluEval QA data (revision-pinned)")
    p.add_argument("--out", type=str, default=str(DEFAULT_RAW_PATH))
    p.add_argument("--url", type=str, default=None)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    try:
        path, sha, size = download_halueval_qa(out_path=Path(args.out), url=args.url, timeout=args.timeout, force=args.force)
        print(f"Saved {path} ({size} bytes), sha256={sha}")
    except Exception as e:
        print("Download failed:", e)
        raise
