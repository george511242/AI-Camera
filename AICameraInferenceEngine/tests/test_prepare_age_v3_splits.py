import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "prepare_age_v3_splits.py"
SPEC = importlib.util.spec_from_file_location("prepare_age_v3_splits", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareAgeV3SplitsTest(unittest.TestCase):
    def test_overlap_counts_checks_ids_and_content(self):
        rows = {
            "train": [{"source_id": "a", "content_sha256": "one"}],
            "validation": [{"source_id": "b", "content_sha256": "two"}],
            "test": [{"source_id": "c", "content_sha256": "one"}],
        }
        result = MODULE.overlap_counts(rows)
        self.assertEqual(result["train_intersection_validation"]["source_id"], 0)
        self.assertEqual(result["train_intersection_test"]["content_sha256"], 1)

    def test_jsonl_is_written_deterministically(self):
        rows = [
            {"source_id": "b", "content_sha256": "two"},
            {"source_id": "a", "content_sha256": "one"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            MODULE.write_jsonl(path, rows)
            written = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["source_id"] for row in written], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
