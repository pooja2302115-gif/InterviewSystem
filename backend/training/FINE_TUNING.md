# Phase 14: Instruction Fine-Tuning

## Pretraining versus supervised fine-tuning

Pretraining teaches the model to continue general corpus text by predicting the next token everywhere in a sequence. Instruction fine-tuning uses examples shaped like:

```json
{
  "instruction": "Explain binary search in simple English.",
  "response": "Binary search finds a target in a sorted sequence..."
}
```

The instruction is context. The response is the supervised target. The model's tokenizer and vocabulary stay the same as the base checkpoint.

## Response-only loss

The formatted input is:

```text
<BOS> Instruction: Explain binary search...
Response: Binary search finds ... <EOS>
```

The dataset marks prompt positions as `False` in `loss_mask` and response tokens plus `<EOS>` as `True`. The existing training loop calculates cross-entropy only on those response positions. This prevents the optimization target from rewarding prompt copying.

## Fine-tuning command

Train a base checkpoint first, then run:

```bash
python backend/training/fine_tune.py \
  --base-checkpoint checkpoints/best_model.pt \
  --train-data backend/data/instruction_train.jsonl \
  --validation-data backend/data/instruction_validation.jsonl \
  --checkpoint-directory checkpoints/instruction \
  --epochs 3 \
  --learning-rate 0.0001
```

The fine-tuning command starts a fresh AdamW optimizer and writes `model_epoch_N.pt` and `best_model.pt` under the instruction checkpoint directory. A smaller learning rate is typical because the base model already contains learned weights.

## Limitations

The seed instruction set contains only a few examples, so it cannot create reliable instruction following. More carefully reviewed examples are needed, and response quality must be checked separately from loss. Fine-tuning also does not add knowledge that was absent from the base model.

## Complexity

Fine-tuning has the same forward/backward cost as pretraining for each batch. The response-only mask reduces the number of loss terms but does not reduce the Transformer forward cost, because the prompt is still needed as context.
