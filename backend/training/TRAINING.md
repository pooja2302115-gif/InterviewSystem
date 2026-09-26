# Phase 12: Training and Validation

## Training flow

For every batch:

```text
input_ids -> InterviewLLM -> logits
                          |
target_ids ----------------+
                          |
              cross-entropy loss
                          |
                    backpropagation
                          |
                       AdamW
```

The model returns logits with shape `(B, L, V)`, where `B` is batch size, `L` is context length, and `V` is vocabulary size. Targets have shape `(B, L)`. Cross-entropy compares the vocabulary scores at every position with the correct next-token ID.

`<PAD>` targets are excluded using `ignore_index=pad_id`, and the dataset's `loss_mask` also defines which positions count. Loss is averaged over valid target tokens rather than padded positions or batches.

## Optimization

The default optimizer is AdamW. After backpropagation, gradient norm clipping limits unusually large updates before `optimizer.step()`. The CLI prints epoch number, training loss, validation loss, and learning rate. Metrics also include token accuracy, valid-token count, and perplexity:

$$
\text{perplexity} = e^{\text{average cross-entropy loss}}
$$

Perplexity is useful for tracking next-token prediction, but it is not a complete measure of chatbot quality.

## Checkpoints

Each epoch writes:

```text
checkpoints/model_epoch_1.pt
checkpoints/model_epoch_2.pt
...
checkpoints/best_model.pt
```

A checkpoint contains model weights, optimizer state, epoch, model configuration, training configuration, and metrics. `best_model.pt` is replaced only when validation loss improves.

## Run training

The processed corpus and vocabulary must exist first:

```bash
python backend/training/prepare_data.py
python backend/model/build_vocab.py
python backend/training/train.py \
  --context-length 64 \
  --embedding-dimension 64 \
  --num-layers 2 \
  --num-heads 4 \
  --feed-forward-dimension 256 \
  --batch-size 2 \
  --epochs 5
```

This is an educational seed corpus. It is too small for useful general-purpose language generation; the training loop proves the mechanics and produces measurements, not production capability.

## Complexity

For $N$ batches, each with $B$ examples and context length $L$, training cost is approximately the model forward/backward cost repeated $N$ times. Attention remains quadratic in $L$. Checkpoint storage is proportional to the number of model and optimizer parameters.
