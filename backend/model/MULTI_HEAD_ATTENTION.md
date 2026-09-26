# Phase 8: Multi-Head Self-Attention

## Why multiple heads

A single attention calculation produces one pattern of token-to-token relationships. Multi-head attention gives the model several smaller attention spaces. It is useful intuition to imagine heads learning different relationships such as syntax, nearby context, or long-range dependencies, but the exact role of a head is learned and must not be assumed in advance.

## Head splitting

For model dimension $d$ and $h$ heads, each head has:

$$
 d_{head} = \frac{d}{h}
$$

The projected tensors are reshaped and transposed:

```text
(batch, sequence, d)
-> (batch, sequence, heads, d_head)
-> (batch, heads, sequence, d_head)
```

Each head computes:

$$
\text{head}_j = \text{softmax}\left(\frac{Q_jK_j^T}{\sqrt{d_{head}}}\right)V_j
$$

The head outputs are transposed back, concatenated along the last dimension, and projected:

```text
(batch, heads, sequence, d_head)
-> (batch, sequence, heads, d_head)
-> (batch, sequence, d)
-> output projection
```

## Masks

The lower-triangular causal mask is broadcast across every head. The Phase 4 padding mask is also broadcast across heads and masks padded keys before softmax. This means no head can read a future or padded key.

## Tensor shapes

For batch size $B$, sequence length $L$, model dimension $d$, and $h$ heads:

```text
Q, K, V:           (B, h, L, d_head)
scores:            (B, h, L, L)
attention weights: (B, h, L, L)
concatenated:      (B, L, d)
output:            (B, L, d)
```

For example, with `(B, L, d, h) = (2, 32, 64, 4)`, each head has dimension 16 and the attention weights have shape `(2, 4, 32, 32)`.

## Configuration constraint

`embedding_dimension` must be divisible by `num_heads`. Uneven heads would make the reshape ambiguous, so the module raises an error instead of silently dropping dimensions.

## Complexity

The score matrices require $O(BhL^2d_{head}) = O(BL^2d)$ time and $O(BhL^2)$ attention-memory space. The quadratic sequence term remains the main cost; multiple heads divide the feature dimension but do not remove that cost.

## Run the tests

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```
