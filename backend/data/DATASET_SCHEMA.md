# Dataset Schema

The initial corpus uses JSON Lines (one JSON object per line). Each record is an independent teaching example that can later be converted into causal language-model text.

## Required fields

- `id`: stable string identifier.
- `category`: broad area such as `dsa`, `programming`, `cs_fundamentals`, or `interview`.
- `topic`: specific concept.
- `question`: prompt shown to the model.
- `answer`: grounded educational response.
- `difficulty`: `beginner`, `intermediate`, or `advanced`.
- `metadata`: searchable tags and source information.

## Optional fields

- `example`: input/output or worked example.
- `code`: source code associated with the answer.
- `language`: programming language for `code`.
- `complexity`: time and space complexity where applicable.

## Dataset rules

1. Keep one valid JSON object per line; do not use comments or trailing commas.
2. Use UTF-8 text and stable IDs so records can be deduplicated later.
3. Keep answers factual and explain assumptions.
4. Do not put secrets, personal student data, or copyrighted books in the dataset.
5. Preserve raw source files under `backend/data/raw/`; cleaned exports belong under `backend/data/processed/`.
6. The train/validation/test files are intentionally tiny seed files. They prove the pipeline shape, not model quality.

## Planned conversion

Phase 2 is implemented by `backend/training/prepare_data.py`. It normalizes prose whitespace, lowercases categories and difficulty values, sorts and deduplicates tags, preserves code line breaks, checks required fields, detects duplicate IDs and rendered content, and writes a cleaning report.

Processed records retain the fields above and add:

- `source_split`: `train`, `validation`, or `test`.
- `text`: a consistent text rendering used as the input to tokenizer development.

In Phase 3, `backend/model/tokenizer.py` tokenizes only the processed training `text` values to build `backend/data/processed/vocab.json`. Validation and test text must not contribute new vocabulary entries. The tokenizer uses `<PAD>`, `<UNK>`, `<BOS>`, and `<EOS>` and produces integer IDs for Phase 4 sequence creation.

Phase 4 uses those IDs to create shifted input/target windows for causal language modeling. The dataset keeps source records separate, pads short windows, and returns attention and loss masks.
