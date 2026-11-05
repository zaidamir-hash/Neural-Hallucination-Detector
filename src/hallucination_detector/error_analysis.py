"""Phase 7: deterministic review export from saved decisions, without inference."""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path

from hallucination_detector.data.acquire import compute_sha256_bytes

MODEL = "mlp"
SEED = 202
LIMIT = 10
CATEGORIES = (
    "entity substitution", "numerical hallucination", "contradiction",
    "high lexical overlap but incorrect meaning", "plausible but unsupported",
    "partial correctness", "context ambiguity", "evidence missed through chunk selection",
    "dataset/label anomaly", "other/unclear",
)
JUDGMENTS = ("model error", "dataset/label anomaly", "unclear judgment")
PROCESSED = "data/processed/halueval_expanded_test.jsonl"
PHASE6 = "results/metrics/phase6/"
OUTPUT = "results/error_analysis/phase7"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def checksum(path: Path) -> str:
    return compute_sha256_bytes(path.read_bytes())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_errors(predictions: list[dict], examples: list[dict], threshold: float,
                  limit: int = LIMIT) -> tuple[list[dict], dict]:
    """Join all original representative predictions before ranking benchmark errors."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1 or limit < 0:
        raise ValueError("Invalid threshold or selection limit")
    chosen = [r for r in predictions if r["model"] == MODEL and str(r["seed"]) == str(SEED)]
    ids = [r["example_id"] for r in examples]
    prediction_ids = [r["example_id"] for r in chosen]
    if (not examples or len(set(ids)) != len(ids) or
            len(set(prediction_ids)) != len(prediction_ids) or set(ids) != set(prediction_ids)):
        raise ValueError("Example IDs must be unique and match the entire test split")
    by_id = {r["example_id"]: r for r in examples}
    matrix = [[0, 0], [0, 0]]
    errors = {"FP": [], "FN": []}
    for prediction in chosen:
        example = by_id[prediction["example_id"]]
        true, predicted = int(prediction["true_label"]), int(prediction["predicted_label"])
        score = float(prediction["continuous_score"])
        if (str(prediction["true_label"]) not in ("0", "1") or
                str(prediction["predicted_label"]) not in ("0", "1") or
                example["label"] != true or example["source_id"] != prediction["source_id"] or
                example["split"] != "test"):
            raise ValueError("Source ID, label or split alignment failure")
        if not math.isfinite(score) or not 0 <= score <= 1 or predicted != int(score >= threshold):
            raise ValueError("Saved score/decision disagrees with locked threshold")
        for key in ("question", "candidate_answer", "reference_context"):
            if not isinstance(example[key], str) or not example[key].strip():
                raise ValueError(f"Missing text: {key}")
        matrix[true][predicted] += 1
        if true == predicted:
            continue
        kind = "FP" if true == 0 else "FN"
        errors[kind].append({
            "example_id": example["example_id"], "source_id": example["source_id"],
            "error_type": kind, "true_label": true, "predicted_label": predicted,
            "hallucination_score": score, "threshold": threshold,
            "score_minus_threshold": score - threshold,
            "absolute_distance": abs(score - threshold),
            "model": MODEL, "seed": SEED, "question": example["question"],
            "candidate_answer": example["candidate_answer"],
            "reference_context": example["reference_context"],
        })
    selected = []
    for kind in ("FP", "FN"):
        ordered = sorted(errors[kind], key=lambda r: (
            -r["hallucination_score"] if kind == "FP" else r["hallucination_score"], r["example_id"]))
        selected.extend(dict(r, rank=rank) for rank, r in enumerate(ordered[:limit], 1))
    return selected, {"confusion_matrix": matrix, "available": {k: len(v) for k, v in errors.items()},
                      "reviewed": {k: min(limit, len(v)) for k, v in errors.items()}}


def attach_reviews(selected: list[dict], notes: list[dict]) -> list[dict]:
    """Require an explicit analyst review and verbatim evidence for every selected case."""
    by_id = {r["example_id"]: r for r in notes}
    if len(by_id) != len(notes) or set(by_id) != {r["example_id"] for r in selected}:
        raise ValueError("Review IDs must exactly match selected IDs, without duplicates")
    reviewed = []
    for row in selected:
        note = by_id[row["example_id"]]
        excerpts, categories = note["context_excerpts"], note["categories"]
        if (not isinstance(excerpts, list) or not excerpts or
                any(not isinstance(e, str) or not e.strip() or e not in row["reference_context"] for e in excerpts) or
                sum(map(len, excerpts)) > 650):
            raise ValueError("Evidence must be concise verbatim context excerpts (at most 650 characters)")
        if (not isinstance(categories, list) or not categories or len(set(categories)) != len(categories) or
                not set(categories) <= set(CATEGORIES)):
            raise ValueError("Invalid review categories")
        if note["judgment"] not in JUDGMENTS or not isinstance(note["analyst_explanation"], str) or not note["analyst_explanation"].strip():
            raise ValueError("Missing analyst judgment/explanation")
        reviewed.append({**{k: v for k, v in row.items() if k != "reference_context"},
                         **{k: note[k] for k in ("context_excerpts", "categories", "judgment", "analyst_explanation")}})
    return reviewed


def render_report(rows: list[dict], manifest: dict, counts: dict) -> str:
    """Render the same reviewed records in a readable report."""
    lines = ["# Phase 7 qualitative error analysis", "", manifest["scope"], "",
             f"Model: MLP seed {SEED}; locked validation threshold: {manifest['threshold']!r}.", "",
             f"Saved confusion matrix [[TN, FP], [FN, TP]]: {manifest['confusion_matrix']}. "
             f"Available errors: {manifest['available']}; reviewed: {manifest['reviewed']}.", "",
             manifest["selection_policy"], "", "## Findings and limits", "",
             "Supported answers are sometimes rejected even when copied from the reference. Several long "
             "grounded answers and short erroneous answers are consistent with the existing style/length "
             "shortcut concern. Short supported false positives also occur. These cases do not identify "
             "which features caused a decision; length is not an explicit primary feature. The already "
             "reported strong answer-only control and shuffled-context results provide separate evidence "
             "of shortcut risk; no experiment was rerun here.", "",
             "Clear date/entity mismatches coexist with apparent label anomalies and insufficient evidence. "
             "Franklin Square is explicitly supported despite label 1; EMC Corporation is supported for Dell EMC "
             "but the question leaves the company unspecified. Brahe is not established as a middle name. "
             "The Curry distance comparison requires geography absent from the reference; the Palmer attribution "
             "has unresolved pronouns. These judgments preserve all benchmark labels. Missing evidence is not "
             "proof of falsehood. No chunk-selection failure is asserted: selected chunks were not inspected.", "",
             "One analyst (the implementing assistant) read each selected question, candidate answer and entire "
             "provided reference context. Only relevant verbatim excerpts are exported. These deliberately selected "
             "extreme-score errors are not representative population estimates. Scores are not calibrated "
             "probabilities or proven confidence. Categories describe evidence and possible failure modes, "
             "not established causal explanations. There was no external fact checking or label revision.", "",
             "## Descriptive counts", "", "Categories allow multiple assignments per case; zero categories are retained.", "",
             "| Category | FP | FN | Total |", "|---|---:|---:|---:|"]
    for category, values in counts["categories"].items():
        lines.append(f"| {category} | {values['FP']} | {values['FN']} | {values['total']} |")
    lines += ["", f"Analyst judgments: {counts['judgments']}", ""]
    for kind, title in (("FP", "False positives: Grounded predicted Hallucinated"),
                        ("FN", "False negatives: Hallucinated predicted Grounded")):
        lines += [f"## {title}", "", "| Rank | Example ID | Score | Score minus threshold | Absolute distance | Judgment |",
                  "|---:|---|---:|---:|---:|---|"]
        for row in rows:
            if row["error_type"] == kind:
                lines.append(f"| {row['rank']} | `{row['example_id']}` | {row['hallucination_score']:.9g} | "
                             f"{row['score_minus_threshold']:+.9g} | {row['absolute_distance']:.9g} | {row['judgment']} |")
        lines.append("")
    lines += ["## Reviewed cases", ""]
    for row in rows:
        lines += [f"### {row['error_type']} {row['rank']}", "",
                  f"- example_id: `{row['example_id']}`", f"- source_id: `{row['source_id']}`",
                  f"- True label: {row['true_label']}; predicted label: {row['predicted_label']} (0=Grounded, 1=Hallucinated).",
                  f"- Hallucination score: {row['hallucination_score']!r}; threshold: {row['threshold']!r}.",
                  f"- Score minus threshold: {row['score_minus_threshold']!r}; absolute distance: {row['absolute_distance']!r}.", "",
                  f"**Question:** {row['question']}", "", f"**Candidate answer:** {row['candidate_answer']}", "",
                  "**Reference evidence:**", ""]
        lines += ["> " + excerpt.replace("\n", "\n> ") + "\n" for excerpt in row["context_excerpts"]]
        lines += [f"**Categories:** {', '.join(row['categories'])}.", "",
                  f"**Judgment:** {row['judgment']}.", "", row["analyst_explanation"], ""]
    return "\n".join(line.rstrip() for line in "\n".join(lines).split("\n"))


def export_phase7(root: Path, notes_path: Path) -> dict:
    """Verify provenance, select fixed errors, validate reviews and export only Phase 7 files."""
    metrics = read_json(root / PHASE6 / "primary_metrics.json")
    lock = read_json(root / PHASE6 / "evaluation_manifest.json")
    validation = read_json(root / "results/metrics/phase5_validation.json")
    saved = read_json(root / PHASE6 / "primary_saved.json")
    # Saved Phase 6 outputs bind the input predictions and metrics to the original evaluation.
    for name, expected in saved["output_checksums"].items():
        if checksum(root / name) != expected:
            raise ValueError(f"Phase 6 artifact checksum mismatch: {name}")
    if checksum(root / PHASE6 / "evaluation_manifest.json") != metrics["evaluation_manifest_checksum"]:
        raise ValueError("Evaluation manifest checksum mismatch")
    if checksum(root / PROCESSED) != metrics["input_checksums"][PROCESSED]:
        raise ValueError("Processed test checksum mismatch")
    validation_path = "results/metrics/phase5_validation.json"
    if checksum(root / validation_path) != lock["input_checksums"][validation_path]:
        raise ValueError("Validation report checksum mismatch")
    if validation["summary"]["representative_seed"] != SEED or metrics["representative_mlp_seed"] != SEED:
        raise ValueError("Representative seed mismatch")
    threshold = next(r["threshold"] for r in validation["runs"] if r["seed"] == SEED)
    result = next(r for r in metrics["results"] if r["model"] == MODEL and r["seed"] == SEED)
    if result["threshold"] != threshold:
        raise ValueError("Locked threshold mismatch")
    with (root / PHASE6 / "primary_predictions.csv").open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    examples = [json.loads(line) for line in (root / PROCESSED).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected, summary = select_errors(predictions, examples, threshold)
    if summary["confusion_matrix"] != result["metrics"]["confusion_matrix"] or len(examples) != metrics["row_count"]:
        raise ValueError("Saved confusion matrix/row count mismatch")
    rows = attach_reviews(selected, read_json(notes_path))
    manifest = {"schema_version": "1.0.0", "model": MODEL, "seed": SEED, "threshold": threshold,
                "limit_per_error_type": LIMIT, **summary,
                "scope": "Original final-test saved predictions only; no inference, training, tuning or changed decisions.",
                "selection_policy": "FP: true 0, predicted 1, score descending. FN: true 1, predicted 0, score ascending. "
                                    "Both: example_id ascending for equal scores; first 10 or all available if fewer. "
                                    "Predicted Hallucinated iff score >= locked validation threshold.",
                "score_interpretation": "Extreme model scores; not calibrated probabilities or proven confidence",
                "selected_example_ids": [r["example_id"] for r in rows],
                "input_checksums": {p: checksum(root / p) for p in (
                    PHASE6 + "primary_predictions.csv", PHASE6 + "primary_metrics.json",
                    PHASE6 + "evaluation_manifest.json", PHASE6 + "primary_saved.json", PROCESSED, validation_path)},
                "analyst_notes_checksum": checksum(notes_path)}
    counts = {"categories": {}, "judgments": dict(Counter(r["judgment"] for r in rows))}
    for category in CATEGORIES:
        c = {kind: sum(category in r["categories"] for r in rows if r["error_type"] == kind) for kind in ("FP", "FN")}
        counts["categories"][category] = {**c, "total": sum(c.values())}
    out = root / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "selection.json", manifest)
    write_json(out / "reviewed_cases.json", rows)
    write_json(out / "category_counts.json", counts)
    (out / "report.md").write_text(render_report(rows, manifest, counts), encoding="utf-8")
    return manifest
