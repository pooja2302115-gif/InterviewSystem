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

    def test_list_definition_returns_only_three_grounded_sections(self):
        answer = self.retriever.answer("What is a Python list?")
        self.assertIn("mutable, ordered sequence", answer)
        self.assertIn("items = [\"bread\", \"milk\"]", answer)
        self.assertEqual(answer.count("**"), 6)
        self.assertNotIn("reduce duplication", answer)

    def test_answers_requested_comparisons_with_situation_based_guidance(self):
        cases = (
            ("Compare list and tuple", ("mutable", "immutable", "fixed record")),
            ("diff binary search and linear search", ("o(log n)", "o(n)", "sorted data")),
            ("For ticket entry, queue or stack which is best?", ("first in, first out", "use a queue", "priority queue")),
        )
        for query, expected_phrases in cases:
            with self.subTest(query=query):
                answer = self.retriever.answer(query)
                for phrase in expected_phrases:
                    self.assertIn(phrase, answer.casefold())

    def test_evaluates_exception_handling_answer(self):
        answer = self.retriever.answer(
            "Evaluate this: An exception is an unexpected event or error that occurs during "
            "the execution of a program and interrupts its normal flow. Exception handling "
            "is used to handle these errors without crashing the program."
        )
        self.assertIn("6.7/10", answer)
        self.assertIn("### Why it is good", answer)
        self.assertIn("### Missing points", answer)
        self.assertIn("### Better interview answer", answer)
        self.assertIn("try/except", answer)

    def test_evaluates_binary_search_with_interview_style_feedback(self):
        answer = self.retriever.answer(
            "Evaluate this: Binary Search is a searching algorithm used to find an element "
            "in a sorted array by repeatedly dividing the search range into two halves."
        )
        self.assertIn("**9/10**", answer)
        self.assertIn("Relevance: relevant.", answer)
        self.assertIn("### Why it is good", answer)
        self.assertIn("### Small improvement", answer)
        self.assertIn("middle element", answer)
        self.assertIn("### Better interview answer", answer)


if __name__ == "__main__":
    unittest.main()
