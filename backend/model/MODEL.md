# Phase 11: Complete Transformer Language Model

## Architecture

`InterviewLLM` connects the completed components:

```text
input token IDs
      |
Token Embedding
      +
Sinusoidal Positional Encoding
      |
Transformer Block 1
      |
Transformer Block 2
      |
...
      |
Transformer Block N
      |
Final LayerNorm
      |
Linear Language-Model Head
      |
logits over vocabulary
```

The model is decoder-only and causal: every block prevents a position from reading future positions. It is not a pretrained model and has no external language-model API dependency.

## Tensor shapes

For vocabulary size $V$, batch size $B$, context length $L$, and model dimension $d$:

```text
input_ids:   (B, L)
embeddings:  (B, L, d)
block output:(B, L, d)
logits:      (B, L, V)
```

Each row of the final logits contains one unnormalized score for every possible next token. Logits are not probabilities; the training loss will apply cross-entropy, which combines a log-softmax with the correct target-token selection.

For example, with vocabulary size 277, batch size 2, context length 32, and model dimension 64, the output shape is `(2, 32, 277)`.

## Padding and masks

If no attention mask is supplied, the model marks IDs different from `padding_idx` as valid. A caller can pass the Phase 4 boolean mask explicitly. The mask prevents padded keys from affecting attention, while the training loss must still ignore padded target IDs using the loss mask or `ignore_index=pad_id`.

## Configuration

The main size controls are configurable in the constructor:

- `vocabulary_size`
- `context_length`
- `embedding_dimension`
- `num_layers`
- `num_heads`
- `feed_forward_dimension`
- `dropout`

A small educational configuration such as `d=64`, `layers=2`, `heads=4`, `context=64` is suitable for quick experiments. Increasing layers, width, context, or vocabulary increases memory and computation substantially.

## Parameter scale

The embedding table and language-model head contribute $Vd$ and approximately $Vd$ parameters. Each Transformer block contributes attention projection parameters plus two feed-forward linear layers and normalization parameters. The exact total is available through `model.parameter_count`.

## Complexity

For $N$ blocks, sequence length $L$, model dimension $d$, and feed-forward dimension $h_{ff}$, the dominant computation is approximately:

$$
O\left(N(BL^2d + BLdh_{ff})\right)
$$

The attention score matrices require $O(NBhL^2)$ memory. This is why context length and model size should remain small while learning.

## Run the tests

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```
