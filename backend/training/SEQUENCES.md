# Phase 4: Training Sequences

## Causal language modeling

The model learns to predict the next token. For token IDs:

```text
full sequence:  <BOS> What is a stack ? <EOS>
input:          <BOS> What is a stack ?
target:         What is a stack ? <EOS>
```

The target is the input shifted one position to the left. At position $t$, the model sees tokens up to $t$ and learns the correct token at $t+1$. The Transformer will later use a causal attention mask so future target tokens are not visible.

## Context windows

`CausalTextDataset` keeps each record separate and divides a long record into non-overlapping windows of `context_length + 1` IDs. Each window produces tensors of length `context_length`:

- `input_ids`: shape `(context_length,)`.
- `target_ids`: shape `(context_length,)`.
- `attention_mask`: boolean shape `(context_length,)`; `True` for real input tokens.
- `loss_mask`: boolean shape `(context_length,)`; `False` for padded targets.

A `DataLoader` stacks these into `(batch_size, context_length)`. Padding uses `<PAD>`. The training loop must either use `loss_mask` or configure cross-entropy with `ignore_index=pad_id`, otherwise padding would be treated as a real target.

## Example

With `context_length=4` and `<BOS> one two <EOS>`, the input is `[BOS, one, two, PAD]` and the target is `[one, two, EOS, PAD]`. The final input and target positions are padding; the third target teaches the model to emit `<EOS>` after `two`.

## Complexity

If the corpus contains $T$ encoded tokens, sequence construction is $O(T)$ time and $O(T)$ storage for the materialized examples. Each training batch uses $O(BL)$ tensor storage, where $B$ is batch size and $L$ is context length.

## Run a real batch

```python
from backend.model.tokenizer import InterviewTokenizer
from backend.training.dataset import create_dataloader

tokenizer = InterviewTokenizer.load("backend/data/processed/vocab.json")
loader = create_dataloader(
    "backend/data/processed/train.jsonl",
    tokenizer,
    context_length=32,
    batch_size=2,
    shuffle=True,
)
batch = next(iter(loader))
print(batch["input_ids"].shape)
```
