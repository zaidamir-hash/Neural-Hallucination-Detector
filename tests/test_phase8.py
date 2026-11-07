"""Small offline checks for published documentation and saved result fidelity."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (ROOT / "README.md", ROOT / "docs/REPRODUCING.md",
             ROOT / "docs/PHASE8_VERIFICATION.md")


def test_documentation_local_links_resolve():
    for document in DOCUMENTS:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            assert (document.parent / target.split("#", 1)[0]).exists(), (document.name, target)


def test_readme_preserves_saved_final_comparison():
    comparison = (ROOT / "results/metrics/phase6/comparison.md").read_text(encoding="utf-8").strip()
    assert comparison in (ROOT / "README.md").read_text(encoding="utf-8")


def test_documented_script_paths_exist():
    for document in DOCUMENTS:
        for script in re.findall(r"scripts/[a-zA-Z0-9_]+\.py", document.read_text(encoding="utf-8")):
            assert (ROOT / script).is_file(), (document.name, script)
