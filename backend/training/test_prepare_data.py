import json
import tempfile
import unittest
from pathlib import Path

from prepare_data import load_and_clean, normalize_text


class PrepareDataTests(unittest.TestCase):
    def test_normalize_text_preserves_code_line_breaks(self):
        value = "  first   line  \nsecond\tline  "
        self.assertEqual(normalize_text(value), "first line\nsecond line")

    def test_cleaner_reports_duplicates_and_malformed_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            record = {
                "id": "one",
                "category": "DSA",
                "topic": "Stack",
                "question": " What is a stack? ",
                "answer": " A stack follows LIFO. ",
                "difficulty": "beginner",
                "metadata": {"tags": [" LIFO ", "data structures"]},
            }
            for split in ("train", "validation", "test"):
                (input_dir / f"{split}.jsonl").write_text("", encoding="utf-8")
            (input_dir / "train.jsonl").write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n{bad\n",
                encoding="utf-8",
            )

            report = load_and_clean(input_dir, output_dir)

            self.assertEqual(report["counts"]["cleaned"], 1)
            self.assertEqual(report["counts"]["duplicate_ids"], 1)
            self.assertEqual(report["counts"]["malformed"], 1)
            processed = (output_dir / "train.jsonl").read_text(encoding="utf-8").strip()
            cleaned = json.loads(processed)
            self.assertEqual(cleaned["category"], "dsa")
            self.assertEqual(cleaned["metadata"]["tags"], ["data structures", "lifo"])
            self.assertIn("Question: What is a stack?", cleaned["text"])


if __name__ == "__main__":
    unittest.main()
