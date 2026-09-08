import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hallucination_detector.data.split_phase2 import (
    assign_group_aware_splits,
    deterministic_example_id,
    expand_records_to_examples,
    load_phase1_validated_records,
    split_config_from_default,
    validate_phase2_leakage,
)


def _synthetic_records():
    # Records 0 and 2 share context after normalization and must stay together.
    return [
        {"original_index": 0, "source_id": "s0", "question": "q0", "knowledge": "CTX A", "right_answer": "r0", "hallucinated_answer": "h0"},
        {"original_index": 1, "source_id": "s1", "question": "q1", "knowledge": "ctx b", "right_answer": "r1", "hallucinated_answer": "h1"},
        {"original_index": 2, "source_id": "s2", "question": "q2", "knowledge": "  ctx    a ", "right_answer": "r2", "hallucinated_answer": "h2"},
        {"original_index": 3, "source_id": "s3", "question": "q3", "knowledge": "ctx c", "right_answer": "r3", "hallucinated_answer": "h3"},
        {"original_index": 4, "source_id": "s4", "question": "q4", "knowledge": "ctx d", "right_answer": "r4", "hallucinated_answer": "h4"},
        {"original_index": 5, "source_id": "s5", "question": "q5", "knowledge": "ctx e", "right_answer": "r5", "hallucinated_answer": "h5"},
    ]


class Phase2UnitTests(unittest.TestCase):
    def test_deterministic_group_aware_split_and_context_grouping(self):
        records = _synthetic_records()
        proportions = {"train": 0.5, "validation": 0.25, "test": 0.25}
        m1 = assign_group_aware_splits(records, proportions, seed=7)
        m2 = assign_group_aware_splits(records, proportions, seed=7)
        self.assertEqual(m1, m2, "Split manifest must be deterministic for fixed seed")

        by_source = {r["source_id"]: r for r in m1}
        self.assertEqual(
            by_source["s0"]["duplicate_context_group_id"],
            by_source["s2"]["duplicate_context_group_id"],
        )
        self.assertEqual(by_source["s0"]["split"], by_source["s2"]["split"])

    def test_split_disjointness_and_manifest_reproducibility(self):
        records = _synthetic_records()
        proportions = {"train": 0.5, "validation": 0.25, "test": 0.25}
        manifest = assign_group_aware_splits(records, proportions, seed=11)
        source_ids = [r["source_id"] for r in manifest]
        self.assertEqual(len(source_ids), len(set(source_ids)))
        split_values = {r["split"] for r in manifest}
        self.assertTrue(split_values.issubset({"train", "validation", "test"}))

    def test_paired_expansion_label_correctness_and_balance(self):
        records = _synthetic_records()
        proportions = {"train": 0.5, "validation": 0.25, "test": 0.25}
        manifest = assign_group_aware_splits(records, proportions, seed=3)
        labels = {"grounded": 0, "hallucinated": 1}
        rows = expand_records_to_examples(records, manifest, labels)
        self.assertEqual(len(rows), len(records) * 2)

        by_source = {}
        for row in rows:
            by_source.setdefault(row["source_id"], []).append(row)
            self.assertIn(row["label"], (0, 1))
            if row["answer_variant"] == "right_answer":
                self.assertEqual(row["label"], 0)
                self.assertEqual(row["label_name"], "Grounded")
            if row["answer_variant"] == "hallucinated_answer":
                self.assertEqual(row["label"], 1)
                self.assertEqual(row["label_name"], "Hallucinated")
            self.assertEqual(
                row["example_id"],
                deterministic_example_id(row["source_id"], row["answer_variant"]),
            )

        for _, pair in by_source.items():
            self.assertEqual(len(pair), 2)
            self.assertEqual({r["split"] for r in pair}.__len__(), 1)

        label_count = {"train": {0: 0, 1: 0}, "validation": {0: 0, 1: 0}, "test": {0: 0, 1: 0}}
        for row in rows:
            label_count[row["split"]][row["label"]] += 1
        for split in label_count:
            self.assertEqual(label_count[split][0], label_count[split][1])

    def test_phase2_leakage_validator(self):
        records = _synthetic_records()
        proportions = {"train": 0.5, "validation": 0.25, "test": 0.25}
        seed = 2
        manifest = assign_group_aware_splits(records, proportions, seed=seed)
        rows = expand_records_to_examples(records, manifest, {"grounded": 0, "hallucinated": 1})
        phase1_manifest = [{"original_index": r["original_index"], "source_id": r["source_id"]} for r in records]
        report = validate_phase2_leakage(
            phase1_manifest=phase1_manifest,
            phase1_exclusions=[],
            split_manifest=manifest,
            expanded_rows=rows,
            split_seed=seed,
            split_proportions=proportions,
            records=records,
        )
        self.assertTrue(report["all_passed"])
        self.assertTrue(report["checks"]["no_exact_duplicate_group_crosses_splits"])

    def test_leakage_validator_detects_context_group_crossing(self):
        records = _synthetic_records()
        proportions = {"train": 0.5, "validation": 0.25, "test": 0.25}
        seed = 2
        manifest = assign_group_aware_splits(records, proportions, seed=seed)
        # Force an invalid crossing: same duplicate-context group across two splits.
        by_sid = {m["source_id"]: m for m in manifest}
        by_sid["s0"]["split"] = "train"
        by_sid["s2"]["split"] = "test"
        bad_manifest = list(by_sid.values())

        rows = expand_records_to_examples(records, bad_manifest, {"grounded": 0, "hallucinated": 1})
        phase1_manifest = [{"original_index": r["original_index"], "source_id": r["source_id"]} for r in records]
        report = validate_phase2_leakage(
            phase1_manifest=phase1_manifest,
            phase1_exclusions=[],
            split_manifest=bad_manifest,
            expanded_rows=rows,
            split_seed=seed,
            split_proportions=proportions,
            records=records,
        )
        self.assertFalse(report["checks"]["no_duplicate_context_group_crosses_splits"])

    def test_invalid_or_incomplete_source_records(self):
        td = Path(tempfile.mkdtemp())
        raw_path = td / "raw.jsonl"
        raw_records = [
            {"question": "q", "knowledge": "k", "right_answer": "r", "hallucinated_answer": "h"},
            {"question": "q2", "knowledge": "k2", "right_answer": "r2"},  # missing field
        ]
        raw_path.write_text("\n".join(json.dumps(r) for r in raw_records), encoding="utf-8")

        manifest_path = td / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                [
                    {"original_index": 0, "source_id": "s0"},
                    {"original_index": 1, "source_id": "s1"},
                ]
            ),
            encoding="utf-8",
        )
        validation_path = td / "validation.json"
        validation_path.write_text(json.dumps({"counts": {"total": 2, "valid": 2, "excluded": 0}, "exclusions": []}), encoding="utf-8")

        with self.assertRaises(ValueError):
            load_phase1_validated_records(raw_path, manifest_path, validation_path)

    def test_configuration_driven_seed_and_proportions(self):
        with patch(
            "hallucination_detector.data.split_phase2.load_config",
            return_value={"split": {"train": 0.7, "validation": 0.15, "test": 0.15, "seed": 42}},
        ):
            proportions, seed = split_config_from_default()
        self.assertEqual(seed, 42)
        self.assertAlmostEqual(proportions["train"], 0.7)
        self.assertAlmostEqual(proportions["validation"], 0.15)
        self.assertAlmostEqual(proportions["test"], 0.15)


if __name__ == "__main__":
    unittest.main()
