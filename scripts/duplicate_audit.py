"""Run duplicate audit for Phase 1."""
from pathlib import Path
import json
from hallucination_detector.data.duplicates import audit_duplicates

RAW = Path("data/raw/qa_data.json")
MANIFEST = Path("data/metadata/halueval_manifest.json")
OUT_JSON = Path("results/metrics/halueval_duplicate_audit.json")
OUT_MD = Path("results/metrics/halueval_duplicate_audit.md")


def _load_raw(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception:
        lines = [l for l in text.splitlines() if l.strip()]
        return [json.loads(l) for l in lines]


def _load_source_ids_by_index(path: Path):
    if not path.exists():
        return None
    items = json.loads(path.read_text(encoding="utf-8"))
    mapping = {}
    for item in items:
        if "original_index" in item and "source_id" in item:
            mapping[int(item["original_index"])] = item["source_id"]
    return mapping


def _render_summary_md(audit):
    rows = [
        ("Exact complete records", "exact_complete_records"),
        ("Normalized questions", "normalized_questions"),
        ("Normalized contexts", "normalized_contexts"),
        ("Normalized question-context pairs", "normalized_question_context_pairs"),
        ("Grounded answers", "grounded_answers"),
        ("Hallucinated answers", "hallucinated_answers"),
    ]
    lines = [
        "# HaluEval QA Duplicate Audit (Phase 1)",
        "",
        "| Category | Duplicate groups | Unique records involved | Largest group size |",
        "|---|---:|---:|---:|",
    ]
    for label, key in rows:
        s = audit["summary"][key]
        lines.append(
            f"| {label} | {s['duplicate_group_count']} | {s['unique_records_involved']} | {s['largest_group_size']} |"
        )
    lines.append("")
    lines.append(
        f"- Invariant `QC -> Q and C duplicate membership`: {audit['invariants']['qc_implies_q_and_k_duplicate_membership']}"
    )
    return "\n".join(lines) + "\n"


def main():
    data = _load_raw(RAW)
    source_ids = _load_source_ids_by_index(MANIFEST)
    out = audit_duplicates(data, source_ids_by_index=source_ids)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(_render_summary_md(out), encoding="utf-8")
    print(f"Duplicate audit written to {OUT_JSON}")
    print(f"Duplicate summary written to {OUT_MD}")


if __name__ == "__main__":
    main()
