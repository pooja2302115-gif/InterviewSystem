import unittest

import torch

from backend.model.transformer_block import TransformerBlock


class TransformerBlockTests(unittest.TestCase):
    def test_output_and_attention_shapes(self):
        block = TransformerBlock(
            embedding_dimension=8,
            num_heads=2,
            feed_forward_dimension=16,
            max_sequence_length=16,
            dropout=0.0,
        )
        output, weights = block(torch.randn(2, 5, 8), return_attention=True)
        self.assertEqual(output.shape, torch.Size([2, 5, 8]))
        self.assertEqual(weights.shape, torch.Size([2, 2, 5, 5]))

    def test_residual_paths_preserve_input_when_sublayers_are_zero(self):
        block = TransformerBlock(embedding_dimension=8, num_heads=2, max_sequence_length=8, dropout=0.0)
        with torch.no_grad():
            for parameter in block.attention.parameters():
                parameter.zero_()
            for parameter in block.feed_forward.parameters():
                parameter.zero_()
        embeddings = torch.randn(2, 5, 8)
        self.assertTrue(torch.allclose(block(embeddings), embeddings))

    def test_attention_mask_reaches_attention_sublayer(self):
        block = TransformerBlock(embedding_dimension=8, num_heads=2, max_sequence_length=8, dropout=0.0)
        _, weights = block(
            torch.randn(1, 4, 8),
            attention_mask=torch.tensor([[True, True, True, False]]),
            return_attention=True,
        )
        self.assertTrue(torch.allclose(weights[:, :, :, 3], torch.zeros(1, 2, 4)))

    def test_gradients_flow_through_both_sublayers(self):
        block = TransformerBlock(embedding_dimension=8, num_heads=2, max_sequence_length=8, dropout=0.0)
        embeddings = torch.randn(2, 4, 8, requires_grad=True)
        block(embeddings).sum().backward()
        self.assertIsNotNone(block.attention.query_projection.weight.grad)
        self.assertIsNotNone(block.feed_forward.input_projection.weight.grad)
        self.assertIsNotNone(embeddings.grad)

    def test_rejects_wrong_embedding_dimension(self):
        block = TransformerBlock(embedding_dimension=8, num_heads=2)
        with self.assertRaises(ValueError):
            block(torch.randn(2, 4, 7))


if __name__ == "__main__":
    unittest.main()
