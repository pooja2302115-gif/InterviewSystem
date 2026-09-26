import tempfile
import unittest
from pathlib import Path

from tokenizer import InterviewTokenizer


class InterviewTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = InterviewTokenizer.build(
            ["What is a stack?", "A stack follows LIFO.\ndef push(item):"],
        )

    def test_special_tokens_have_stable_ids(self):
        self.assertEqual(self.tokenizer.pad_id, 0)
        self.assertEqual(self.tokenizer.unk_id, 1)
        self.assertEqual(self.tokenizer.bos_id, 2)
        self.assertEqual(self.tokenizer.eos_id, 3)

    def test_tokenize_and_decode_prose(self):
        tokens = self.tokenizer.tokenize("What is a stack?")
        self.assertEqual(tokens, ["What", "is", "a", "stack", "?"])
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode("What is a stack?")), "What is a stack?")

    def test_unknown_tokens_and_padding(self):
        encoded = self.tokenizer.encode(
            "What is a queue?",
            max_length=8,
            padding=True,
            truncation=True,
        )
        self.assertIn(self.tokenizer.unk_id, encoded)
        self.assertEqual(len(encoded), 8)
        self.assertEqual(encoded[-1], self.tokenizer.pad_id)

    def test_truncation_retains_end_marker(self):
        encoded = self.tokenizer.encode("one two three four", max_length=4, truncation=True)
        self.assertEqual(encoded[0], self.tokenizer.bos_id)
        self.assertEqual(encoded[-1], self.tokenizer.eos_id)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.json"
            self.tokenizer.save(path)
            restored = InterviewTokenizer.load(path)
            self.assertEqual(restored.token_to_id, self.tokenizer.token_to_id)


if __name__ == "__main__":
    unittest.main()
