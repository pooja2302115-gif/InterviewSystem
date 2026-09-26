# AI Interview Preparation System

An educational interview-preparation system built around a small Transformer language model implemented with PyTorch. The model will be developed from scratch for learning purposes; no pretrained language model or hosted LLM API is part of the design.

## Phase 1: Project structure and dataset design

This phase establishes the boundaries for the later implementation:

```text
.
├── backend/
│   ├── chatbot/          # Conversation and inference logic (later phase)
│   ├── data/
│   │   ├── raw/          # Untouched source material
│   │   ├── processed/    # Cleaned/generated data
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   ├── model/            # Tokenizer and Transformer modules (later phases)
│   ├── resume/           # Resume parsing and analysis (later phase)
│   └── training/         # Dataset, training, and evaluation code (later phases)
├── checkpoints/          # Saved model weights; ignored by Git
├── frontend/             # React application (later phase)
├── requirements.txt
└── README.md
```

## Dataset design

The seed corpus is JSONL: one JSON object per line. It includes examples from:

- DSA: stacks, binary search, hash maps, and later linked lists and other topics.
- Programming: a Python implementation example with complexity metadata.
- CS fundamentals: DBMS, operating systems, networks, and OOP.
- Interviews: project explanations, behavioral answers, resume questions, and HR preparation.

The complete record contract is documented in [backend/data/DATASET_SCHEMA.md](backend/data/DATASET_SCHEMA.md). The current corpus is intentionally small. It validates the shape of the pipeline, but it is far too small to produce a capable conversational model.

## Why JSONL first?

JSONL lets us keep each training example independently readable, validate records line by line, add metadata without changing the file format, and stream larger datasets later. In Phase 2, the records will be cleaned and rendered into consistent text examples. In later phases, those texts will become token IDs and causal language-model input/target pairs.

## Install and validate

Create an isolated environment and install the planned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Validate the seed data without training anything:

```bash
python - <<'PY'
import json
from pathlib import Path

required = {"id", "category", "topic", "question", "answer", "difficulty", "metadata"}
ids = set()
count = 0
for path in sorted(Path("backend/data").glob("*.jsonl")):
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        missing = required - record.keys()
        assert not missing, f"{path}:{line_number} missing {missing}"
        assert record["id"] not in ids, f"duplicate id: {record['id']}"
        ids.add(record["id"])
        count += 1
print(f"Validated {count} records with {len(ids)} unique IDs.")
PY
```

## Important limitation

A from-scratch model trained on a small personal dataset will not have the knowledge, reasoning ability, or reliability of a production-scale model. This repository prioritizes understanding the data and model mechanics. Evaluation, hallucination checks, and honest scope limits will remain part of every later phase.

## Phase 2: Cleaning and normalization

Run the cleaner with:

```bash
python backend/training/prepare_data.py
```

It reads the three seed splits, validates required fields, normalizes prose and metadata, preserves code formatting, removes duplicate IDs or duplicate rendered records across splits, and writes processed JSONL files plus `cleaning_report.json` under `backend/data/processed/`. A malformed or invalid record is reported and causes a non-zero exit status.

Run its focused tests with:

```bash
python -m unittest discover -s backend/training -p 'test_*.py'
```

## Next phase

## Phase 3: Tokenizer from scratch

The tokenizer is implemented in [backend/model/tokenizer.py](backend/model/tokenizer.py). It uses a small, inspectable regular-expression tokenizer that separates words, numbers, punctuation, and newlines. It does not use a pretrained vocabulary or a tokenizer library.

Build its vocabulary from the training split only:

```bash
python backend/model/build_vocab.py
```

The tokenizer supports:


The four reserved IDs are deterministic: `<PAD>=0`, `<UNK>=1`, `<BOS>=2`, and `<EOS>=3`. The vocabulary is sorted by frequency and then token text, which makes builds reproducible. The current tokenizer is intentionally word-and-symbol based; it is easy to understand but less efficient than a subword tokenizer and may produce many `<UNK>` tokens on unseen code or words.

Run focused tests with:

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```

## Phase 4: Training sequences

The causal dataset is implemented in [backend/training/dataset.py](backend/training/dataset.py). It converts each processed record into shifted `input_ids` and `target_ids`, then pads windows to a configurable `context_length`. It also returns an `attention_mask` for real input tokens and a `loss_mask` so padding does not contribute to training loss.

The detailed tensor shapes and next-token example are documented in [backend/training/SEQUENCES.md](backend/training/SEQUENCES.md). Tests can be run with:

```bash
python -m unittest discover -s backend/training -p 'test_*.py'
```

## Phase 5: Token embeddings

The token embedding layer is implemented in [backend/model/embeddings.py](backend/model/embeddings.py). `TokenEmbedding` maps each integer token ID to a trainable vector using PyTorch's `nn.Embedding`, preserves batch and sequence dimensions, and keeps the `<PAD>` row fixed at zero.

The concepts, tensor shapes, parameter count, and complexity are documented in [backend/model/EMBEDDINGS.md](backend/model/EMBEDDINGS.md). Run the model tests with:

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```

## Phase 6: Sinusoidal positional encoding

The positional encoding module is implemented in [backend/model/positional_encoding.py](backend/model/positional_encoding.py). It adds deterministic sine and cosine vectors to token embeddings, so attention can distinguish token order without introducing trainable parameters.

The formula, tensor shapes, complexity, and sequence-length behavior are documented in [backend/model/POSITIONAL_ENCODING.md](backend/model/POSITIONAL_ENCODING.md). Run the model tests with:

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```

## Phase 7: Causal self-attention

The single-head scaled dot-product attention module is implemented in [backend/model/attention.py](backend/model/attention.py). It computes query, key, and value projections, scales $QK^T$, applies the causal lower-triangular mask, optionally masks padded keys, and returns context-aware representations.

The attention equations, tensor shapes, mask behavior, and complexity are documented in [backend/model/ATTENTION.md](backend/model/ATTENTION.md). Run the model tests with:

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```

## Phase 8: Multi-head self-attention

The multi-head implementation is in [backend/model/multi_head_attention.py](backend/model/multi_head_attention.py). It splits the model dimension into equal heads, applies the Phase 7 causal scaled dot-product operation independently, concatenates the head outputs, and applies a final projection.

The reshape steps, tensor shapes, mask behavior, divisibility rule, and complexity are documented in [backend/model/MULTI_HEAD_ATTENTION.md](backend/model/MULTI_HEAD_ATTENTION.md). Run the model tests with:

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```

## Phase 9: Feed-forward network

The position-wise feed-forward network is implemented in [backend/model/feed_forward.py](backend/model/feed_forward.py). It applies `Linear -> GELU -> Dropout -> Linear` to each sequence position, using a configurable hidden dimension that defaults to four times the embedding dimension.

The equations, tensor shapes, parameter count, and complexity are documented in [backend/model/FEED_FORWARD.md](backend/model/FEED_FORWARD.md). Run the model tests with:

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```

## Phase 10: Transformer block

The pre-layer-normalized Transformer block is implemented in [backend/model/transformer_block.py](backend/model/transformer_block.py). It applies layer normalization, multi-head causal self-attention, a residual connection, a second layer normalization, the feed-forward network, and a second residual connection.

The architecture, equations, tensor shapes, configuration, and complexity are documented in [backend/model/TRANSFORMER_BLOCK.md](backend/model/TRANSFORMER_BLOCK.md). Run the model tests with:

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```

## Phase 11: Complete Transformer language model

The complete decoder-only model is implemented in [backend/model/model.py](backend/model/model.py) as `InterviewLLM`. It combines token embeddings, sinusoidal positions, a configurable stack of Transformer blocks, final layer normalization, and a linear language-model head.

For input shape `(batch, sequence)`, it returns logits with shape `(batch, sequence, vocabulary_size)`. The logits are unnormalized next-token scores; Phase 12 will use them with cross-entropy during training.

The architecture, masks, configuration, parameter scale, tensor shapes, and complexity are documented in [backend/model/MODEL.md](backend/model/MODEL.md). Run the model tests with:

```bash
python -m unittest discover -s backend/model -p 'test_*.py'
```

## Phase 12: Training and validation

The training pipeline is implemented in [backend/training/train.py](backend/training/train.py), with settings in [backend/training/config.py](backend/training/config.py). It uses cross-entropy with padding ignored, AdamW, gradient clipping, validation loss, token accuracy, perplexity, and epoch/best-model checkpoints.

The training flow, loss masking, metrics, checkpoint contents, and command are documented in [backend/training/TRAINING.md](backend/training/TRAINING.md). Run the focused tests with:

```bash
python -m unittest discover -s backend/training -p 'test_*.py'
```

## Phase 13: Held-out evaluation

The standalone evaluator is implemented in [backend/training/evaluate.py](backend/training/evaluate.py). It reloads a saved checkpoint, reconstructs its model configuration, evaluates the held-out test split, and reports loss, perplexity, token accuracy, and valid-token count without updating weights.

The metric definitions, test-set isolation rules, command, and limitations are documented in [backend/training/EVALUATION.md](backend/training/EVALUATION.md). Run the focused tests with:

```bash
python -m unittest discover -s backend/training -p 'test_*.py'
```

## Phase 14: Instruction fine-tuning

Instruction-response data is stored in [backend/data/instruction_train.jsonl](backend/data/instruction_train.jsonl) and [backend/data/instruction_validation.jsonl](backend/data/instruction_validation.jsonl), with its schema documented in [backend/data/INSTRUCTION_SCHEMA.md](backend/data/INSTRUCTION_SCHEMA.md). [instruction_dataset.py](backend/training/instruction_dataset.py) masks prompt tokens so supervised loss applies only to response tokens and `<EOS>`.

The fine-tuning command is [fine_tune.py](backend/training/fine_tune.py). Its data format, response-only loss, command, and limitations are documented in [FINE_TUNING.md](backend/training/FINE_TUNING.md). Run the focused tests with:

```bash
python -m unittest discover -s backend/training -p 'test_*.py'
```

## Phase 15: Inference and conversation handling

Inference is implemented in [backend/chatbot/inference.py](backend/chatbot/inference.py), with session history in [conversation.py](backend/chatbot/conversation.py). The custom tokenizer and checkpoint-backed `InterviewLLM` support greedy decoding, temperature/top-k sampling, `<EOS>` stopping, context-window trimming, and bounded session history.

The generation flow, conversation contract, usage example, and limitations are documented in [CHATBOT.md](backend/chatbot/CHATBOT.md). Run chatbot tests with:

```bash
python -m unittest discover -s backend/chatbot -p 'test_*.py'
```

## Phase 16: FastAPI backend

The API is implemented in [backend/app.py](backend/app.py). It exposes `/health`, `POST /chat`, session-history retrieval, and session deletion, with CORS configured for the future React client. Model loading is lazy/configurable through `INTERVIEW_CHECKPOINT` and `INTERVIEW_VOCABULARY`.

Endpoint contracts, startup instructions, examples, and current in-memory-session limitations are documented in [API.md](backend/API.md). Run API tests with:

```bash
python -m unittest backend.test_app
```

Start the development server after configuring a checkpoint:

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Phase 17 will build the React chatbot interface.
Phase 10 will combine normalization, residual connections, multi-head attention, and this feed-forward network into a Transformer block.
