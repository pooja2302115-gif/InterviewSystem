# Phase 6: Sinusoidal Positional Encoding

## Why position is needed

Self-attention compares token vectors, but it does not inherently know whether a token came first, second, or later. The sequences `I love Python` and `Python love I` contain the same words but have different meanings because their order differs.

The model therefore receives:

```text
 token embedding + position encoding
```

The token embedding describes what a token is; the positional vector describes where it occurs in the context window.

## Sinusoidal formula

For position $p$ and embedding dimension index $i$:

$$
PE_{p, 2i} = \sin\left(\frac{p}{10000^{2i/d}}\right)
$$

$$
PE_{p, 2i+1} = \cos\left(\frac{p}{10000^{2i/d}}\right)
$$

Here, $d$ is the embedding dimension. Different dimensions use different wavelengths. Short wavelengths vary quickly across nearby positions, while long wavelengths change more slowly and provide broader position information.

At position zero, every sine component is 0 and every cosine component is 1. Position vectors are deterministic and do not add trainable parameters.

## Tensor shapes

For embeddings with shape `(batch_size, sequence_length, embedding_dimension)`, the registered encoding buffer has shape `(1, max_sequence_length, embedding_dimension)`. Broadcasting adds the same position vector to every item in the batch, producing the original embedding shape.

For example:

```text
embeddings: (2, 32, 64)
encoding:   (1, 256, 64)
result:     (2, 32, 64)
```

The module rejects a sequence longer than `max_sequence_length`; choose a larger configured value instead of silently losing position information.

## Complexity

Precomputing the encoding table costs $O(Md)$ time and memory, where $M$ is the maximum sequence length and $d$ is the embedding dimension. Adding it to a batch costs $O(BLd)$ time for batch size $B$ and sequence length $L$.

## Run the tests

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```
