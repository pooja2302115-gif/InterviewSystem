# Phase 3: Tokenizer

## Why tokenization is needed

A Transformer receives numbers, not raw text. The tokenizer creates a finite vocabulary and maps each token to an integer:

```text
text -> token strings -> token IDs -> embedding lookup
```

For example, this tokenizer produces a readable sequence such as:

```text
What is a stack?
-> ["What", "is", "a", "stack", "?"]
```

The exact IDs depend on the vocabulary learned from the training split. IDs are not meanings; they are indexes into the embedding table that will be built in a later phase.

## Tokenization strategy

`InterviewTokenizer` is intentionally simple and transparent:

- Words and identifiers are tokens, including names such as `two_sum`.
- Numbers and decimal values are tokens.
- Punctuation and operators are individual tokens.
- Newlines are explicit tokens so multi-line code is not flattened completely.
- Unknown words map to `<UNK>` instead of changing the vocabulary during inference.

This is not a subword tokenizer. It is easier to study, but rare words and unseen identifiers can produce more `<UNK>` tokens. A later educational improvement can add subword pieces without changing the Transformer interface: the model still receives integer IDs.

## Special tokens

| Token | ID | Purpose |
|---|---:|---|
| `<PAD>` | 0 | Fills shorter examples in a batch. Its loss contribution will be masked later. |
| `<UNK>` | 1 | Represents a token absent from the training vocabulary. |
| `<BOS>` | 2 | Marks the beginning of a sequence. |
| `<EOS>` | 3 | Marks the end of a sequence. |

Encoding adds `<BOS>` and `<EOS>` by default. With `max_length` and `padding=True`, sequences are padded to the same length. With `truncation=True`, long sequences are shortened and the final token remains `<EOS>`.

## Reproducible vocabulary

The builder reads only `backend/data/processed/train.jsonl`. Validation and test text must not add vocabulary entries because that would leak information across evaluation boundaries. Tokens are ordered by descending frequency and then lexicographically, so rebuilding with the same training data gives the same IDs.

```bash
python backend/model/build_vocab.py
```

The result is `backend/data/processed/vocab.json`.

## Complexity

For a text containing $n$ characters, regex tokenization is $O(n)$ for this pattern. Building the vocabulary over $N$ tokens is $O(N \log V)$ because frequencies are sorted, where $V$ is the number of distinct tokens. Encoding and decoding are $O(L)$ for a sequence of $L$ tokens or IDs. The vocabulary uses $O(V)$ memory.

These costs are appropriate for the educational corpus. Production tokenizers use more sophisticated algorithms and optimized implementations for large datasets.
