import unittest

import torch

from backend.model.positional_encoding import SinusoidalPositionalEncoding


class SinusoidalPositionalEncodingTests(unittest.TestCase):
    def test_output_shape_and_position_zero_values(self):
        encoding = SinusoidalPositionalEncoding(embedding_dimension=6, max_sequence_length=8)
        output = encoding(torch.zeros(2, 4, 6))
        self.assertEqual(output.shape, torch.Size([2, 4, 6]))
        self.assertTrue(torch.equal(output[0, 0, 0::2], torch.zeros(3)))
        self.assertTrue(torch.equal(output[0, 0, 1::2], torch.ones(3)))

    def test_positions_are_different(self):
        encoding = SinusoidalPositionalEncoding(embedding_dimension=6, max_sequence_length=8)
        output = encoding(torch.zeros(1, 4, 6))
        self.assertFalse(torch.equal(output[0, 0], output[0, 1]))

    def test_supports_odd_embedding_dimensions(self):
        encoding = SinusoidalPositionalEncoding(embedding_dimension=5)
        output = encoding(torch.zeros(1, 2, 5))
        self.assertEqual(output.shape, torch.Size([1, 2, 5]))

    def test_encoding_is_a_non_trainable_buffer(self):
        encoding = SinusoidalPositionalEncoding(embedding_dimension=6)
        self.assertEqual(list(encoding.parameters()), [])
        self.assertIn("encoding", dict(encoding.named_buffers()))

    def test_rejects_too_long_sequences(self):
        encoding = SinusoidalPositionalEncoding(embedding_dimension=6, max_sequence_length=4)
        with self.assertRaises(ValueError):
            encoding(torch.zeros(1, 5, 6))


if __name__ == "__main__":
    unittest.main()
