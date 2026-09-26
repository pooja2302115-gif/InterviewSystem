import json
import tempfile
import unittest
from pathlib import Path

import torch

from backend.model.model import InterviewLLM
from backend.model.tokenizer import InterviewTokenizer
from backend.training.config import TrainingConfig
from backend.training.dataset import create_dataloader
from backend.training.evaluate import evaluate_checkpoint, load_checkpoint_model
from backend.training.train import train_model


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = InterviewTokenizer.build(["one two three four five"])
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        data_path = Path(self.directory.name) / "data.jsonl"
        records = [{"id": "one", "text": "one two three four"}, {"id": "two", "text": "two three four five"}]
        data_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        self.loader = create_dataloader(data_path, self.tokenizer, context_length=4, batch_size=2, shuffle=False)

    def make_model(self):
        return InterviewLLM(
            vocabulary_size=self.tokenizer.vocabulary_size,
            context_length=4,
            embedding_dimension=8,
            num_layers=1,
            num_heads=2,
            feed_forward_dimension=16,
            dropout=0.0,
        )

    def test_load_and_evaluate_checkpoint(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as checkpoint_directory:
            checkpoint_path = Path(checkpoint_directory) / "best_model.pt"
            train_model(
                model,
                self.loader,
                self.loader,
                pad_id=self.tokenizer.pad_id,
                config=TrainingConfig(epochs=1, checkpoint_directory=Path(checkpoint_directory), device="cpu"),
            )
            loaded_model, checkpoint = load_checkpoint_model(checkpoint_path, device=torch.device("cpu"))
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(loaded_model.context_length, 4)
            report = evaluate_checkpoint(
                checkpoint_path,
                self.loader,
                pad_id=self.tokenizer.pad_id,
                device=torch.device("cpu"),
            )
            self.assertEqual(report["epoch"], 1)
            self.assertGreater(report["tokens"], 0)
            self.assertGreater(report["perplexity"], 0)

    def test_missing_checkpoint_is_clear(self):
        with self.assertRaises(FileNotFoundError):
            load_checkpoint_model("missing.pt", device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
