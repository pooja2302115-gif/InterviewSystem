# Phase 7: Causal Self-Attention

## What self-attention does

Each token creates a query, key, and value from its embedding:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

A query measures how relevant every key is to the current token. The raw similarity matrix is scaled dot product attention:

$$
S = \frac{QK^T}{\sqrt{d_k}}
$$

The scale prevents large dot products from making softmax gradients too small. Softmax turns each score row into weights, and the weighted values produce the context-aware representation:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

This Phase 7 module uses one attention head. Phase 8 will split the same operation into multiple heads.

## Causal masking

For next-token prediction, position $t$ must not read positions greater than $t$. The lower-triangular mask is:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

A zero means the score is replaced with a very negative value before softmax, making its probability effectively zero. The module also accepts the Phase 4 boolean `attention_mask` to prevent padded keys from being attended to.

The mask applies to keys. A padded query can still produce an output from earlier valid tokens, but its output must be ignored by the loss mask during training.

## Tensor shapes

For input embeddings with shape `(batch, sequence, embedding_dimension)`:

```text
Q, K, V:          (batch, 1, sequence, embedding_dimension)
scores:           (batch, 1, sequence, sequence)
attention weights:(batch, sequence, sequence)
output:           (batch, sequence, embedding_dimension)
```

The explicit single-head dimension keeps the transition to multi-head attention visible in the next phase.

## Small numerical intuition

If a query is more similar to key 2 than key 1, its softmax weights give value 2 more influence. The output is not a copied token; it is a weighted sum of value vectors. With the causal mask, a query at position 1 can use positions 0 and 1, but never position 2.

## Complexity

For sequence length $L$ and embedding dimension $d$, forming the score matrix costs $O(L^2d)$ time and $O(L^2)$ attention-memory space per batch item. The quadratic $L^2$ term is why context length is an important model configuration.

## Run the tests

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```
