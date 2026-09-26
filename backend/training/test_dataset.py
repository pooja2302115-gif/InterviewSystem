import json
import tempfile
import unittest
from pathlib import Path

import torch

from backend.model.tokenizer import InterviewTokenizer
from backend.training.dataset import CausalTextDataset, create_dataloader


class CausalTextDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = InterviewTokenizer.build(["one two three four five"])

    def make_data(self, text="one two"):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "data.jsonl"
        path.write_text(json.dumps({"id": "sample", "text": text}) + "\n", encoding="utf-8")
        self.addCleanup(directory.cleanup)
        return path

    def test_targets_are_shifted_one_token(self):
        dataset = CausalTextDataset(self.make_data("one two three four five"), self.tokenizer, context_length=4)
        example = dataset[0]
        self.assertTrue(torch.equal(example["target_ids"][:-1], example["input_ids"][1:]))
        self.assertEqual(example["target_ids"][-1].item(), self.tokenizer.token_to_id["four"])
        self.assertEqual(example["input_ids"].shape, torch.Size([4]))
        self.assertEqual(example["target_ids"].shape, torch.Size([4]))

    def test_padding_masks_are_false_after_valid_tokens(self):
        dataset = CausalTextDataset(self.make_data("one"), self.tokenizer, context_length=4)
        example = dataset[0]
        self.assertEqual(example["attention_mask"].tolist(), [True, True, False, False])
        self.assertEqual(example["loss_mask"].tolist(), [True, True, False, False])
        self.assertEqual(example["target_ids"][-1].item(), self.tokenizer.pad_id)

    def test_dataloader_batches_tensors(self):
        loader = create_dataloader(
            self.make_data("one two"),
            self.tokenizer,
            context_length=4,
            batch_size=1,
            shuffle=False,
        )
        batch = next(iter(loader))
        self.assertEqual(batch["input_ids"].shape, torch.Size([1, 4]))
        self.assertEqual(batch["input_ids"].dtype, torch.long)
        self.assertEqual(batch["attention_mask"].dtype, torch.bool)
        self.assertEqual(batch["record_id"], ["sample"])


if __name__ == "__main__":
    unittest.main()
