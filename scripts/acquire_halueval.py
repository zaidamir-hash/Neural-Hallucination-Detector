"""Script entry point to download HaluEval QA data (Phase 1).

Usage:
  python scripts/acquire_halueval.py --out data/raw/qa_data.json --force
"""
from pathlib import Path
from hallucination_detector.data.acquire import (
    download_halueval_qa,
    DEFAULT_COMMIT,
    utc_timestamp_z,
)
import json
import hashlib
import argparse

PROV_DIR = Path("data/metadata")
PROV_DIR.mkdir(parents=True, exist_ok=True)
PROV_FILE = PROV_DIR / "halueval_qa_provenance.json"


def main():
    parser = argparse.ArgumentParser(description="Acquire official HaluEval QA dataset.")
    parser.add_argument("--out", type=str, default="data/raw/qa_data.json")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_path = Path(args.out)

    try:
        path, sha, size = download_halueval_qa(
            out_path=out_path,
            timeout=args.timeout,
            force=args.force,
        )
    except FileExistsError:
        print(f"Raw file already exists at {out_path}. Computing provenance from existing file.")
        path = out_path
        raw_text = path.read_bytes()
        sha = hashlib.sha256(raw_text).hexdigest()
        size = path.stat().st_size
    except Exception as e:
        print("Download failed:", e)
        raise

    # compute record count (support JSON array or NDJSON)
    raw_text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text)
    except Exception:
        lines = [l for l in raw_text.splitlines() if l.strip()]
        data = [json.loads(l) for l in lines]
    record_count = len(data)

    provenance = {
        "dataset_name": "HaluEval",
        "subset": "qa",
        "official_repo": "https://github.com/RUCAIBox/HaluEval",
        "download_url": f"https://raw.githubusercontent.com/RUCAIBox/HaluEval/{DEFAULT_COMMIT}/data/qa_data.json",
        "source_revision": DEFAULT_COMMIT,
        "download_date": utc_timestamp_z(),
        "local_raw_path": str(path).replace("\\", "/"),
        "sha256": sha,
        "raw_file_size": size,
        "raw_record_count": record_count,
        "required_fields": ["question", "knowledge", "right_answer", "hallucinated_answer"],
        "license": "MIT",
        "paper_or_url": "https://arxiv.org/abs/2305.11747",
    }

    PROV_FILE.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Downloaded {path} ({record_count} records), provenance written to {PROV_FILE}")


if __name__ == "__main__":
    main()
