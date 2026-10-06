import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from backend.chatbot.conversation import ConversationManager
from backend.chatbot.inference import GenerationConfig, InferenceEngine
from backend.chatbot.retrieval import QuestionBankRetriever
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
        self.assertTrue(generator.prompts[1].startswith("Instruction:"))
        self.assertTrue(generator.prompts[1].endswith("Response:"))

    def test_history_is_bounded(self):
        manager = ConversationManager(FakeGenerator(), max_history_turns=1)
        session_id = manager.chat("one")["session_id"]
        manager.chat("two", session_id)
        self.assertEqual(len(manager.get_history(session_id)), 2)

    def test_empty_message_is_rejected(self):
        with self.assertRaises(ValueError):
            ConversationManager(FakeGenerator()).chat(" ")

    def test_known_python_definition_uses_curated_answer(self):
        generator = FakeGenerator()
        manager = ConversationManager(generator, retriever=QuestionBankRetriever())
        result = manager.chat("[Technical interview] define Python")
        self.assertIn("programming language", result["response"])
        self.assertEqual(generator.prompts, [])


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

    def test_checkpoint_vocabulary_mismatch_reports_sizes_and_recovery(self):
        tokenizer = InterviewTokenizer.build(["hello", "world"])
        model = InterviewLLM(
            tokenizer.vocabulary_size + 1,
            context_length=8,
            embedding_dimension=8,
            num_layers=1,
            num_heads=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            vocabulary_path = Path(directory) / "vocab.json"
            tokenizer.save(vocabulary_path)
            with patch("backend.chatbot.inference.load_checkpoint_model", return_value=(model, {})):
                with self.assertRaisesRegex(ValueError, "uses .* tokens.*contains .*Use the vocabulary saved"):
                    InferenceEngine.from_checkpoint("model.pt", vocabulary_path, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
