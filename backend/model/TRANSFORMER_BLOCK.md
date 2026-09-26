# Phase 10: Transformer Block

## Architecture

The block combines the components built so far using pre-layer normalization:

```text
input x
  |
LayerNorm
  |
Multi-Head Causal Self-Attention
  |
residual: x + attention_output
  |
LayerNorm
  |
Feed-Forward Network
  |
residual: residual + feed_forward_output
  |
output
```

Pre-norm means normalization happens before each sublayer. The residual connections give the block a direct path for information and gradients. If either sublayer learns that no change is useful, it can produce values near zero and the residual path preserves the incoming representation.

## Equations

For input $x$:

$$
 y = x + \text{Attention}(\text{LayerNorm}(x))
$$

$$
 z = y + \text{FFN}(\text{LayerNorm}(y))
$$

The attention module still applies causal and padding-key masks, and the block returns its per-head attention weights when requested.

## Tensor shapes

For batch size $B$, sequence length $L$, and model dimension $d$:

```text
input:             (B, L, d)
attention output:  (B, L, d)
feed-forward output:(B, L, d)
block output:      (B, L, d)
```

The hidden feed-forward dimension may be larger internally, but the residual addition requires the sublayer output to return to $d$.

## Configuration

The block exposes:

- `embedding_dimension`: model width.
- `num_heads`: attention head count; it must divide the model width.
- `feed_forward_dimension`: hidden width, defaulting to four times the model width.
- `max_sequence_length`: causal-mask capacity.
- `dropout`: dropout used by attention and the feed-forward network.
- `layer_norm_epsilon`: numerical stability constant for normalization.

## Complexity

The attention sublayer dominates with $O(BL^2d)$ time and $O(BhL^2)$ attention memory. The feed-forward sublayer adds $O(BLdh_{ff})$ time, where $h_{ff}$ is its hidden dimension.

## Run the tests

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```
