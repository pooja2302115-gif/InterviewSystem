import unittest

import torch

from backend.model.multi_head_attention import MultiHeadSelfAttention


class MultiHeadSelfAttentionTests(unittest.TestCase):
    def test_output_and_attention_shapes(self):
        attention = MultiHeadSelfAttention(
            embedding_dimension=8,
            num_heads=2,
            max_sequence_length=16,
        )
        output, weights = attention(torch.randn(3, 5, 8), return_attention=True)
        self.assertEqual(output.shape, torch.Size([3, 5, 8]))
        self.assertEqual(weights.shape, torch.Size([3, 2, 5, 5]))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(3, 2, 5)))

    def test_future_positions_are_masked_for_every_head(self):
        attention = MultiHeadSelfAttention(embedding_dimension=8, num_heads=2, max_sequence_length=4)
        values = torch.randn(1, 4, 8)
        _, weights = attention(values, return_attention=True)
        self.assertTrue(torch.allclose(weights.triu(1), torch.zeros_like(weights.triu(1))))

        changed_future = values.clone()
        changed_future[:, 3] += 1000
        self.assertTrue(torch.allclose(attention(values)[:, 0], attention(changed_future)[:, 0]))

    def test_padding_keys_are_masked_for_every_head(self):
        attention = MultiHeadSelfAttention(embedding_dimension=8, num_heads=2, max_sequence_length=4)
        _, weights = attention(
            torch.randn(1, 4, 8),
            attention_mask=torch.tensor([[True, True, True, False]]),
            return_attention=True,
        )
        self.assertTrue(torch.allclose(weights[:, :, :, 3], torch.zeros(1, 2, 4)))

    def test_embedding_dimension_must_divide_into_heads(self):
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(embedding_dimension=7, num_heads=2)

    def test_head_dimension_is_exposed(self):
        attention = MultiHeadSelfAttention(embedding_dimension=12, num_heads=3)
        self.assertEqual(attention.head_dimension, 4)


if __name__ == "__main__":
    unittest.main()
