# Instruction Fine-Tuning Data

Instruction records use JSONL with one object per line:

- `id`: stable record identifier.
- `instruction`: task or question shown as the prompt.
- `response`: target answer written by a human or carefully reviewed source.
- `category`: task type such as `explain`, `compare`, `code`, `debug`, `complexity`, `mock_interview`, `hr`, or `resume`.
- `metadata`: tags and source information.

The Phase 14 dataset covers explanations, comparisons, code, debugging, complexity, mock interviews, HR answers, and resume-based questions. It is intentionally small and educational.

During supervised fine-tuning, the prompt is included in the model input so the model can condition on it, but only response tokens and `<EOS>` contribute to the loss. This is implemented by `InstructionDataset.loss_mask`. Prompt-only loss would teach the model to copy the instruction instead of focusing on producing a response.

The tokenizer and vocabulary remain the same as the base model. Do not rebuild a separate vocabulary for instruction data unless the model's embedding and language-model head are intentionally resized.
