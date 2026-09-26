# Phase 15: Inference and Conversation Handling

## Autoregressive generation

Inference starts with a prompt, encodes it with the custom tokenizer, and repeatedly predicts one next token:

```text
prompt IDs
   |
InterviewLLM
   |
last-position logits
   |
selection: greedy or temperature/top-k sampling
   |
append token and repeat
```

Generation stops at `<EOS>` or `max_new_tokens`. When the prompt exceeds the model context, the most recent context is retained. The model is always put in evaluation mode and generation runs without gradients.

`InferenceEngine` supports deterministic greedy decoding with `temperature=0` and probabilistic decoding with positive temperature. `top_k` restricts sampling to the k highest-scoring tokens. These controls affect decoding only; they do not improve the tiny model's learned knowledge.

## Conversation sessions

`ConversationManager` stores user and assistant messages under a session ID. It creates a UUID when no session ID is supplied, includes bounded history in the next prompt, and trims older turns to control context growth. The session response has the requested shape:

```json
{
  "response": "...",
  "session_id": "..."
}
```

The current prompt contains explicit `System`, `User`, and `Assistant` labels. Later API work can wrap this manager in a `/chat` endpoint.

## Usage

```python
import torch
from backend.chatbot.inference import InferenceEngine, GenerationConfig

engine = InferenceEngine.from_checkpoint(
    "checkpoints/instruction/best_model.pt",
    "backend/data/processed/vocab.json",
    device=torch.device("cpu"),
)
answer = engine.generate_text(
    "Explain binary search.",
    config=GenerationConfig(max_new_tokens=40, temperature=0.0),
)
print(answer)
```

The checkpoint must exist first. A randomly initialized or minimally trained checkpoint will produce poor responses; this implementation demonstrates the mechanics honestly.
