import unittest

import torch

from backend.model.attention import CausalSelfAttention


class CausalSelfAttentionTests(unittest.TestCase):
    def make_identity_attention(self):
        attention = CausalSelfAttention(embedding_dimension=2, max_sequence_length=4)
        with torch.no_grad():
            for projection in (
                attention.query_projection,
                attention.key_projection,
                attention.value_projection,
                attention.output_projection,
            ):
                projection.weight.copy_(torch.eye(2))
                projection.bias.zero_()
        attention.eval()
        return attention

    def test_output_and_attention_shapes(self):
        attention = CausalSelfAttention(embedding_dimension=8, max_sequence_length=16)
        output, weights = attention(torch.randn(2, 5, 8), return_attention=True)
        self.assertEqual(output.shape, torch.Size([2, 5, 8]))
        self.assertEqual(weights.shape, torch.Size([2, 5, 5]))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 5)))

    def test_future_positions_are_masked(self):
        attention = self.make_identity_attention()
        values = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]])
        changed_future = values.clone()
        changed_future[0, 2] = torch.tensor([100.0, 100.0])
        first_output, weights = attention(values, return_attention=True)
        second_output = attention(changed_future)
        self.assertTrue(torch.allclose(first_output[:, 0], second_output[:, 0]))
        self.assertTrue(torch.allclose(weights[0].triu(1), torch.zeros(3, 3)))

    def test_padding_keys_are_masked(self):
        attention = self.make_identity_attention()
        values = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]])
        _, weights = attention(
            values,
            attention_mask=torch.tensor([[True, True, False]]),
            return_attention=True,
        )
        self.assertTrue(torch.allclose(weights[0, :, 2], torch.zeros(3)))

    def test_rejects_non_boolean_attention_mask(self):
        attention = CausalSelfAttention(embedding_dimension=2)
        with self.assertRaises(TypeError):
            attention(torch.randn(1, 2, 2), attention_mask=torch.ones(1, 2))

    def test_rejects_too_long_sequence(self):
        attention = CausalSelfAttention(embedding_dimension=2, max_sequence_length=2)
        with self.assertRaises(ValueError):
            attention(torch.randn(1, 3, 2))


if __name__ == "__main__":
    unittest.main()
