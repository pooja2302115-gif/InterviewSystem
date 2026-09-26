import unittest

import torch

from backend.model.tokenizer import InterviewTokenizer
from backend.training.instruction_dataset import InstructionDataset


class InstructionDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = InterviewTokenizer.load("backend/data/processed/vocab.json")
        cls.dataset = InstructionDataset(
            "backend/data/instruction_train.jsonl",
            cls.tokenizer,
            context_length=64,
        )

    def test_loads_instruction_records(self):
        self.assertEqual(len(self.dataset), 8)

    def test_prompt_is_context_but_not_loss_target(self):
        example = self.dataset[0]
        loss_positions = example["loss_mask"].nonzero().flatten()
        self.assertGreater(len(loss_positions), 0)
        first_response_position = int(loss_positions[0])
        self.assertTrue(torch.all(~example["loss_mask"][:first_response_position]))
        self.assertTrue(torch.all(example["attention_mask"][:first_response_position]))

    def test_tensors_have_fixed_shapes(self):
        example = self.dataset[0]
        self.assertEqual(example["input_ids"].shape, torch.Size([64]))
        self.assertEqual(example["target_ids"].shape, torch.Size([64]))
        self.assertEqual(example["attention_mask"].dtype, torch.bool)
        self.assertEqual(example["loss_mask"].dtype, torch.bool)

    def test_padding_does_not_contribute_to_loss(self):
        example = self.dataset[0]
        padding_positions = ~example["attention_mask"]
        self.assertTrue(torch.all(~example["loss_mask"][padding_positions]))


if __name__ == "__main__":
    unittest.main()
