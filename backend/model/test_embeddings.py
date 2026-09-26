import unittest

import torch

from backend.model.embeddings import TokenEmbedding


class TokenEmbeddingTests(unittest.TestCase):
    def test_lookup_preserves_batch_and_sequence_dimensions(self):
        embedding = TokenEmbedding(vocabulary_size=12, embedding_dimension=8)
        input_ids = torch.tensor([[0, 2, 4], [3, 5, 0]], dtype=torch.long)
        output = embedding(input_ids)
        self.assertEqual(output.shape, torch.Size([2, 3, 8]))

    def test_padding_row_is_zero_and_has_no_gradient(self):
        embedding = TokenEmbedding(vocabulary_size=12, embedding_dimension=8, padding_idx=0)
        input_ids = torch.tensor([[0, 2, 4]], dtype=torch.long)
        output = embedding(input_ids)
        self.assertTrue(torch.equal(output[0, 0], torch.zeros(8)))
        output.sum().backward()
        self.assertTrue(torch.equal(embedding.embedding.weight.grad[0], torch.zeros(8)))
        self.assertGreater(embedding.embedding.weight.grad[2].abs().sum().item(), 0)

    def test_parameter_count(self):
        embedding = TokenEmbedding(vocabulary_size=12, embedding_dimension=8)
        self.assertEqual(embedding.parameter_count, 96)

    def test_rejects_invalid_ids_dtype(self):
        embedding = TokenEmbedding(vocabulary_size=12, embedding_dimension=8)
        with self.assertRaises(TypeError):
            embedding(torch.tensor([[1.0]]))


if __name__ == "__main__":
    unittest.main()
