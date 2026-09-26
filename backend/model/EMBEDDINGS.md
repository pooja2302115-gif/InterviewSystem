# Phase 5: Token Embeddings

## What an embedding does

Token IDs are category indexes, not useful numeric measurements. An embedding layer stores one trainable vector per vocabulary entry:

```text
ID 0 (<PAD>) -> vector of length d
ID 1 (<UNK>) -> vector of length d
ID 2 (<BOS>) -> vector of length d
...
```

For a vocabulary size $V$ and embedding dimension $d$, the embedding table is a matrix $E \in \mathbb{R}^{V \times d}$. Looking up token ID $i$ returns row $E_i$:

$$
\text{embedding}(i) = E_i
$$

The vectors begin as learned parameters and are updated by backpropagation. Tokens that occur in similar contexts may develop useful geometric relationships, but this tiny educational model is not large enough to guarantee meaningful semantics.

## Tensor shapes

For `input_ids` with shape `(batch_size, sequence_length)`, `TokenEmbedding` returns:

```text
(batch_size, sequence_length, embedding_dimension)
```

With vocabulary size 277, batch size 2, context length 32, and embedding dimension 64:

```text
input_ids:  (2, 32)
embeddings: (2, 32, 64)
```

The sequence and batch dimensions are preserved because each integer ID is replaced by one vector.

## Padding behavior

`padding_idx` is set to the tokenizer's `<PAD>` ID, which is 0. The padding vector is explicitly initialized to zeros. PyTorch also prevents gradients from updating this row. This keeps padded positions from gaining a learned meaning; the attention and loss masks from Phase 4 still need to be used later.

## Parameter count and complexity

The layer has $V \times d$ trainable values. For $V=277$ and $d=64$, that is 17,728 parameters. A lookup for $L$ IDs copies $L$ rows and costs $O(Ld)$ time and output memory; the full table uses $O(Vd)$ memory.

## Run the tests

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```
