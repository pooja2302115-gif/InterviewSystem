import unittest

import torch

from backend.model.model import InterviewLLM


class InterviewLLMTests(unittest.TestCase):
    def make_model(self):
        return InterviewLLM(
            vocabulary_size=20,
            context_length=8,
            embedding_dimension=16,
            num_layers=2,
            num_heads=4,
            feed_forward_dimension=32,
            dropout=0.0,
        )

    def test_logits_have_expected_shape(self):
        model = self.make_model()
        logits = model(torch.tensor([[2, 5, 7, 0], [2, 4, 0, 0]], dtype=torch.long))
        self.assertEqual(logits.shape, torch.Size([2, 4, 20]))

    def test_returns_one_attention_map_per_layer(self):
        model = self.make_model()
        logits, attention_maps = model(
            torch.tensor([[2, 5, 7, 0]], dtype=torch.long),
            return_attention=True,
        )
        self.assertEqual(logits.shape, torch.Size([1, 4, 20]))
        self.assertEqual(len(attention_maps), 2)
        self.assertEqual(attention_maps[0].shape, torch.Size([1, 4, 4, 4]))

    def test_automatic_padding_mask_masks_padded_keys(self):
        model = self.make_model()
        _, attention_maps = model(
            torch.tensor([[2, 5, 7, 0]], dtype=torch.long),
            return_attention=True,
        )
        for attention in attention_maps:
            self.assertTrue(torch.allclose(attention[:, :, :, 3], torch.zeros(1, 4, 4)))

    def test_gradients_reach_embeddings_and_language_model_head(self):
        model = self.make_model()
        input_ids = torch.tensor([[2, 5, 7, 0]], dtype=torch.long)
        model(input_ids).mean().backward()
        self.assertIsNotNone(model.token_embedding.embedding.weight.grad)
        self.assertIsNotNone(model.language_model_head.weight.grad)

    def test_rejects_sequence_longer_than_context(self):
        model = self.make_model()
        with self.assertRaises(ValueError):
            model(torch.ones(1, 9, dtype=torch.long))

    def test_parameter_count_is_positive(self):
        self.assertGreater(self.make_model().parameter_count, 0)


if __name__ == "__main__":
    unittest.main()
