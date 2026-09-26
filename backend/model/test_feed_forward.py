import unittest

import torch

from backend.model.feed_forward import FeedForwardNetwork


class FeedForwardNetworkTests(unittest.TestCase):
    def test_preserves_batch_and_sequence_shape(self):
        network = FeedForwardNetwork(embedding_dimension=8)
        output = network(torch.randn(2, 5, 8))
        self.assertEqual(output.shape, torch.Size([2, 5, 8]))

    def test_default_hidden_dimension_is_four_times_model_dimension(self):
        network = FeedForwardNetwork(embedding_dimension=8)
        self.assertEqual(network.hidden_dimension, 32)

    def test_custom_hidden_dimension_and_parameter_count(self):
        network = FeedForwardNetwork(embedding_dimension=8, hidden_dimension=16)
        expected = (8 * 16 + 16) + (16 * 8 + 8)
        self.assertEqual(network.parameter_count, expected)

    def test_position_wise_processing_uses_same_network(self):
        network = FeedForwardNetwork(embedding_dimension=4, hidden_dimension=8)
        network.eval()
        one_position = torch.randn(1, 1, 4)
        repeated_positions = one_position.repeat(1, 3, 1)
        output = network(repeated_positions)
        self.assertTrue(torch.allclose(output[:, 0], output[:, 1]))
        self.assertTrue(torch.allclose(output[:, 1], output[:, 2]))

    def test_rejects_wrong_embedding_dimension(self):
        network = FeedForwardNetwork(embedding_dimension=8)
        with self.assertRaises(ValueError):
            network(torch.randn(2, 5, 7))


if __name__ == "__main__":
    unittest.main()
