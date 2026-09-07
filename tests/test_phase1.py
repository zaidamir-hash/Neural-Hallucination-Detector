import unittest
import tempfile
from pathlib import Path
import json
from datetime import datetime
from hallucination_detector.data.acquire import download_halueval_qa, utc_timestamp_z
from hallucination_detector.data.validate import validate_records, compute_source_id
from hallucination_detector.data.duplicates import audit_duplicates, normalize_text

class Phase1UnitTests(unittest.TestCase):
    def test_download_refuses_overwrite(self):
        # create a temp file where download would go
        td = Path(tempfile.mkdtemp())
        target = td / "qa_data.json"
        target.write_text('[]', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            download_halueval_qa(out_path=target, timeout=1, force=False)

    def test_validation_required_field_checks(self):
        # create synthetic NDJSON-like content
        records = [
            {"question":"q1","knowledge":"k1","right_answer":"r1","hallucinated_answer":"h1"},
            {"question":"q2","knowledge":"k2","right_answer":None,"hallucinated_answer":"h2"},
            {"question":"q3","knowledge":"k3","right_answer":"r3","hallucinated_answer":""},
            "not_an_object",
        ]
        # write to temp file
        td = Path(tempfile.mkdtemp())
        p = td / "raw.json"
        p.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
        annotated, counts, exclusions = validate_records(p)
        self.assertEqual(counts['total'], 4)
        # expect at least two exclusions (null field and empty field and non-object)
        self.assertTrue(counts['excluded'] >= 2)
        reasons = [r for _, r in exclusions]
        self.assertTrue(any(r.startswith("null_field:") for r in reasons))
        self.assertTrue(any(r.startswith("empty_field:") for r in reasons))
        self.assertIn("non_object", reasons)

    def test_identical_answer_detection(self):
        records = [
            {
                "question": "q",
                "knowledge": "k",
                "right_answer": "same  answer",
                "hallucinated_answer": "same answer",
            }
        ]
        td = Path(tempfile.mkdtemp())
        p = td / "raw.json"
        p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        _, counts, exclusions = validate_records(p)
        self.assertEqual(counts["excluded"], 1)
        self.assertEqual(exclusions[0][1], "identical_answers")

    def test_source_id_deterministic_and_sensitive(self):
        r1 = {"question":"q","knowledge":"k","right_answer":"r","hallucinated_answer":"h"}
        r2 = {"question":"q","knowledge":"k","right_answer":"r","hallucinated_answer":"h"}
        id1 = compute_source_id(r1)
        id2 = compute_source_id(r2)
        self.assertEqual(id1, id2)
        r3 = {"question":"q!","knowledge":"k","right_answer":"r","hallucinated_answer":"h"}
        id3 = compute_source_id(r3)
        self.assertNotEqual(id1, id3)

    def test_duplicate_normalization_consistency_and_invariant(self):
        recs = [
            {"question": "Q  one", "knowledge": "K one", "right_answer": "R", "hallucinated_answer": "H"},
            {"question": " q one ", "knowledge": "k   one ", "right_answer": "R2", "hallucinated_answer": "H2"},
            {"question": "Q two", "knowledge": "K two", "right_answer": "R3", "hallucinated_answer": "H3"},
            {"question": "Q two", "knowledge": "K two", "right_answer": "R4", "hallucinated_answer": "H4"},
        ]
        out = audit_duplicates(recs)
        self.assertTrue(out["invariants"]["qc_implies_q_and_k_duplicate_membership"])
        self.assertEqual(normalize_text(" Q   one "), normalize_text("q one"))

    def test_duplicate_audit_compact_hashed_groups_and_summary(self):
        recs = [
            {"question": "q", "knowledge": "k", "right_answer": "ra", "hallucinated_answer": "ha"},
            {"question": "q ", "knowledge": "k ", "right_answer": "ra", "hallucinated_answer": "ha2"},
            {"question": "q2", "knowledge": "k2", "right_answer": "ra2", "hallucinated_answer": "ha2"},
            {"question": "q2", "knowledge": "k2", "right_answer": "ra3", "hallucinated_answer": "ha3"},
        ]
        source_map = {i: f"id-{i}" for i in range(len(recs))}
        out = audit_duplicates(recs, source_ids_by_index=source_map)

        # compact: categories are list-based and hashed
        for cat_name, groups in out["categories"].items():
            self.assertIsInstance(groups, list, cat_name)
            for g in groups:
                self.assertIn("group_key_hash", g)
                self.assertEqual(len(g["group_key_hash"]), 64)
                self.assertIn("group_size", g)
                self.assertIn("original_indices", g)
                self.assertIn("source_ids", g)
                self.assertTrue(all(isinstance(i, int) for i in g["original_indices"]))
                # hash key should not contain full text
                self.assertNotIn("knowledge", g["group_key_hash"].lower())

        # exact summary checks for one category
        q_summary = out["summary"]["normalized_questions"]
        self.assertEqual(q_summary["duplicate_group_count"], 2)
        self.assertEqual(q_summary["unique_records_involved"], 4)
        self.assertEqual(q_summary["largest_group_size"], 2)

    def test_provenance_timestamp_timezone_aware(self):
        ts = utc_timestamp_z()
        self.assertTrue(ts.endswith("Z"))
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        self.assertIsNotNone(parsed.tzinfo)
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)

    def test_provenance_required_fields_schema(self):
        required = {
            "dataset_name",
            "subset",
            "official_repo",
            "download_url",
            "source_revision",
            "download_date",
            "local_raw_path",
            "sha256",
            "raw_file_size",
            "raw_record_count",
            "required_fields",
            "license",
            "paper_or_url",
        }
        td = Path(tempfile.mkdtemp())
        prov = td / "prov.json"
        sample = {k: "x" for k in required}
        sample["required_fields"] = ["question", "knowledge", "right_answer", "hallucinated_answer"]
        sample["raw_file_size"] = 123
        sample["raw_record_count"] = 2
        prov.write_text(json.dumps(sample), encoding="utf-8")
        loaded = json.loads(prov.read_text(encoding="utf-8"))
        self.assertTrue(required.issubset(set(loaded.keys())))

if __name__ == '__main__':
    unittest.main()
