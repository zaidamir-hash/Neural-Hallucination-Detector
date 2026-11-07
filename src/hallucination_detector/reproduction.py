"""Create an independent workspace from an explicit set of inputs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


WORKSPACE = "reproduction_workspace.json"
REFERENCE = "reference/phase7"
OUTPUT_DIRS = ("data/raw", "data/metadata", "data/splits", "data/processed",
               "data/embedding_cache", "data/features", "data/scaler",
               "checkpoints", "results/metrics", "results/figures", "notebooks")


def source_commit(root: Path) -> str:
    """Use the captured source HEAD for workspaces without Git metadata."""
    marker = root / WORKSPACE
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))["source_commit"]
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True).stdout.strip()


def initialize_workspace(source: Path, target: Path) -> None:
    """Copy code and explicit reference inputs, leaving generated outputs empty."""
    copies = {}
    for directory, pattern in (("src", "*.py"), ("scripts", "*.py"), ("tests", "*.py"),
                               ("configs", "*.yaml"), ("configs", "*.json")):
        for path in (source / directory).rglob(pattern):
            copies[path.relative_to(source)] = path
    for name in ("README.md", "PROJECT_PLAN.md", "AGENTS.md", "pyproject.toml",
                 "requirements.txt", "constraints-tested.txt", ".gitignore"):
        copies[Path(name)] = source / name
    copies[Path("reference/halueval_qa_provenance.json")] = source / "data/metadata/halueval_qa_provenance.json"
    for name in ("analyst_notes.json", "reviewed_cases.json"):
        copies[Path(REFERENCE) / name] = source / "results/error_analysis/phase7" / name

    commit = source_commit(source)
    target.mkdir()
    hashes = {}
    for relative, original in sorted(copies.items()):
        contents = original.read_bytes()
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
        hashes[relative.as_posix()] = hashlib.sha256(contents).hexdigest()
    for directory in OUTPUT_DIRS:
        (target / directory).mkdir(parents=True, exist_ok=True)
    (target / WORKSPACE).write_text(json.dumps({
        "source_commit": commit, "copied_input_sha256": hashes,
    }, indent=2) + "\n", encoding="utf-8")
