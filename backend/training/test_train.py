import json
import tempfile
import unittest
from pathlib import Path

import torch

from backend.model.model import InterviewLLM
from backend.model.tokenizer import InterviewTokenizer
from backend.training.config import TrainingConfig
from backend.training.dataset import create_dataloader
from backend.training.train import evaluate_model, train_model


class TrainingTests(unittest.TestCase):
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

    def test_evaluation_returns_loss_perplexity_and_accuracy(self):
        metrics = evaluate_model(self.make_model(), self.loader, pad_id=self.tokenizer.pad_id, device=torch.device("cpu"))
        self.assertGreater(metrics["loss"], 0)
        self.assertGreater(metrics["perplexity"], 1)
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)
        self.assertGreater(metrics["tokens"], 0)

    def test_training_writes_epoch_and_best_checkpoints(self):
        model = self.make_model()
        before = model.language_model_head.weight.detach().clone()
        with tempfile.TemporaryDirectory() as checkpoint_directory:
            history = train_model(
                model,
                self.loader,
                self.loader,
                pad_id=self.tokenizer.pad_id,
                config=TrainingConfig(epochs=1, checkpoint_directory=Path(checkpoint_directory), device="cpu"),
            )
            epoch_checkpoint = Path(checkpoint_directory) / "model_epoch_1.pt"
            best_checkpoint = Path(checkpoint_directory) / "best_model.pt"
            self.assertEqual(len(history), 1)
            self.assertTrue(epoch_checkpoint.exists())
            self.assertTrue(best_checkpoint.exists())
            payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["epoch"], 1)
            self.assertIn("model_state_dict", payload)
            self.assertFalse(torch.equal(before, model.language_model_head.weight.detach()))


if __name__ == "__main__":
    unittest.main()
