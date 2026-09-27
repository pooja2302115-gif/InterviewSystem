import unittest

from backend.chatbot.retrieval import QuestionBankRetriever


class QuestionBankRetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = QuestionBankRetriever()

    def test_retrieves_python_definition(self):
        answer = self.retriever.answer("[Technical interview] define Python")
        self.assertIsNotNone(answer)
        self.assertIn("programming language", answer)
        self.assertNotEqual(answer, "A.")

    def test_retrieves_formal_definition_when_requested(self):
        answer = self.retriever.answer("Give a formal textbook definition of Python")
        self.assertIn("interpreted", answer)
        self.assertIn("general-purpose", answer)

    def test_coding_match_includes_solution_and_complexity(self):
        answer = self.retriever.answer("Write Python code for two sum and give time complexity")
        self.assertIn("def two_sum", answer)
        self.assertIn("O(n)", answer)
        self.assertIn("Example output", answer)

    def test_generates_requested_full_stack_question_count(self):
        answer = self.retriever.answer(
            "Generate 20 questions skills are python, html, css and job role full stack developer"
        )
        questions = [line for line in answer.splitlines() if line[:1].isdigit() and ". " in line]
        self.assertEqual(len(questions), 20)
        self.assertIn("Python", answer)
        self.assertIn("HTML", answer)
        self.assertIn("CSS", answer)

    def test_unrelated_question_returns_no_match(self):
        self.assertIsNone(self.retriever.answer("What is the capital of an imaginary planet?"))


if __name__ == "__main__":
    unittest.main()
