# Expanded Curated Corpus

The current dataset is generated reproducibly by `generate_corpus.py` from curated topic notes plus structured Python-fundamentals, cross-subject, and coding-interview records. It currently contains 122 distinct topic labels and covers DSA, Python/Java/C/C++, databases and SQL, operating systems, networking, OOP, software testing, computer architecture, web APIs, cloud, AI, security, HR interviewing, projects, and resume preparation. See `QUESTION_BANK_SCHEMA.md` for the richer format.

Current split sizes:

| File | Records | Purpose |
|---|---:|---|
| `train.jsonl` | 314 | Base causal language-model training |
| `validation.jsonl` | 33 | Checkpoint selection during base training |
| `test.jsonl` | 32 | Held-out base-model evaluation |
| `instruction_train.jsonl` | 426 | Response-only supervised fine-tuning |
| `instruction_validation.jsonl` | 36 | Fine-tuning validation |

Each topic is assigned wholly to one split before question variants are created. This avoids putting near-identical questions about the same topic into both training and evaluation splits. The tokenizer vocabulary is built from base and instruction training data only; neither validation nor test data contributes vocabulary tokens.

## Regenerate and prepare

From the repository root:

```bash
python backend/data/generate_corpus.py
python backend/training/prepare_data.py
python backend/model/build_vocab.py
```

`generate_corpus.py` writes the three base splits and two instruction splits from `TOPICS`, `python_fundamentals.jsonl`, `coding_questions.jsonl`, and `cross_subject_questions.jsonl`. Add reviewed records to those sources; do not hand-edit only generated base splits if you need changes to survive regeneration.

## Train

```bash
python backend/training/train.py \
  --checkpoint-directory checkpoints \
  --context-length 128 \
  --embedding-dimension 64 \
  --num-layers 2 \
  --num-heads 4 \
  --feed-forward-dimension 256 \
  --batch-size 8 \
  --epochs 3 \
  --device cpu

python backend/training/fine_tune.py \
  --base-checkpoint checkpoints/best_model.pt \
  --checkpoint-directory checkpoints/instruction \
  --batch-size 8 \
  --epochs 3 \
  --learning-rate 0.0001 \
  --device cpu
```

Rebuilding the vocabulary changes the model vocabulary size. Retrain the base model before fine-tuning; checkpoints created with an older vocabulary are incompatible.

## Honest scope

This is hundreds of curated records and question variants, not a large-scale language-model corpus. It is suitable for demonstrating the full educational pipeline and small experiments, but not enough for reliable broad CS knowledge or production-quality answers. A genuinely large dataset needs thousands to millions of legally usable, reviewed examples, deduplication, provenance tracking, and compute appropriate to the resulting model. Do not inflate the apparent dataset size by repeating the same answer with cosmetic prompt changes.
