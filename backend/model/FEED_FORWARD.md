# Phase 9: Feed-Forward Network

## Purpose

Attention mixes information between sequence positions. The feed-forward network then transforms each position independently and adds nonlinear capacity:

```text
embedding at one position
        |
      Linear
        |
      GELU
        |
      Dropout
        |
      Linear
        |
updated representation at the same position
```

The same weights are reused at every position. It is therefore position-wise: it does not compare token 1 with token 2. Cross-position interaction comes from attention.

## Equations

For input vector $x$:

$$
\text{FFN}(x) = W_2\,\text{GELU}(W_1x + b_1) + b_2
$$

The hidden dimension is commonly $4d$, where $d$ is the model embedding dimension. GELU is a smooth activation that allows the network to model nonlinear relationships.

## Tensor shapes

For input shape `(batch_size, sequence_length, embedding_dimension)`:

```text
input:  (B, L, d)
hidden: (B, L, hidden_dimension)
output: (B, L, d)
```

With `d=64` and the default hidden dimension, the hidden tensor has shape `(B, L, 256)`. The batch and sequence dimensions are unchanged.

## Parameter count and complexity

With model dimension $d$ and hidden dimension $h$, the two linear layers contain:

$$
(dh + h) + (hd + d)
$$

parameters, including biases. The computation costs $O(BLdh)$ time and uses $O(BLh)$ hidden activation memory.

## Run the tests

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```
