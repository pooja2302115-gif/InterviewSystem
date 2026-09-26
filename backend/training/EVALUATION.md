# Phase 13: Held-Out Evaluation

## Why a separate test split

Training loss measures memorization and validation loss helps choose settings. The held-out test split should be read only after training and model selection are complete. It estimates how the saved checkpoint behaves on examples that did not guide parameter updates or checkpoint selection.

The evaluator loads `best_model.pt`, reconstructs the model from the checkpoint's `model_config`, and runs the test JSONL through the same padding-aware metric calculation used during validation.

## Metrics

- **Loss:** average cross-entropy over valid target tokens.
- **Perplexity:** $e^{loss}$, the model's effective uncertainty in next-token prediction.
- **Token accuracy:** fraction of valid positions whose highest-scoring token matches the target. This can be misleading for language because many valid continuations exist.
- **Valid tokens:** number of non-padding target positions included in the report.

A low loss or high token accuracy does not prove that responses are helpful, factual, coherent, or safe. The seed corpus is far too small for meaningful chatbot claims.

## Run evaluation

Train a checkpoint first:

```bash
python backend/training/train.py --epochs 5
```

Then evaluate only the held-out test split:

```bash
python backend/training/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --test-data backend/data/processed/test.jsonl \
  --vocabulary backend/data/processed/vocab.json
```

The command prints JSON similar to:

```json
{
  "checkpoint": "checkpoints/best_model.pt",
  "epoch": 5,
  "loss": 5.2,
  "perplexity": 181.0,
  "accuracy": 0.08,
  "tokens": 700
}
```
