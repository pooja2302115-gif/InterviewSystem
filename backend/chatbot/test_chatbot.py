import tempfile
import unittest
from pathlib import Path

import torch

from backend.chatbot.conversation import ConversationManager
from backend.chatbot.inference import GenerationConfig, InferenceEngine
from backend.model.model import InterviewLLM
from backend.model.tokenizer import InterviewTokenizer


class FakeGenerator:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "Practice response."


class ConversationTests(unittest.TestCase):
    def test_chat_creates_session_and_keeps_history(self):
        generator = FakeGenerator()
        manager = ConversationManager(generator, max_history_turns=2)
        first = manager.chat("What is a stack?")
        second = manager.chat("What is its complexity?", first["session_id"])
        self.assertEqual(first["session_id"], second["session_id"])
        self.assertEqual(len(manager.get_history(first["session_id"])), 4)
        self.assertIn("What is a stack?", generator.prompts[1])

    def test_history_is_bounded(self):
        manager = ConversationManager(FakeGenerator(), max_history_turns=1)
        session_id = manager.chat("one")["session_id"]
        manager.chat("two", session_id)
        self.assertEqual(len(manager.get_history(session_id)), 2)

    def test_empty_message_is_rejected(self):
        with self.assertRaises(ValueError):
            ConversationManager(FakeGenerator()).chat(" ")


class InferenceTests(unittest.TestCase):
    def test_generation_returns_at_most_requested_tokens(self):
        tokenizer = InterviewTokenizer.build(["Instruction: hello Response: answer"])
        model = InterviewLLM(
            tokenizer.vocabulary_size,
            context_length=8,
            embedding_dimension=8,
            num_layers=1,
            num_heads=2,
            feed_forward_dimension=16,
            dropout=0.0,
        )
        engine = InferenceEngine(model, tokenizer, device=torch.device("cpu"))
        generated = engine.generate_token_ids("hello", config=GenerationConfig(max_new_tokens=3))
        self.assertLessEqual(len(generated), 3)

    def test_generation_config_rejects_invalid_values(self):
        tokenizer = InterviewTokenizer.build(["hello"])
        model = InterviewLLM(tokenizer.vocabulary_size, context_length=8, embedding_dimension=8, num_layers=1, num_heads=2)
        engine = InferenceEngine(model, tokenizer, device=torch.device("cpu"))
        with self.assertRaises(ValueError):
            engine.generate_text("hello", config=GenerationConfig(max_new_tokens=0))


if __name__ == "__main__":
    unittest.main()
