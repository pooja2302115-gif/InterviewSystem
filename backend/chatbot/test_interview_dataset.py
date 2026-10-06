import unittest

from backend.chatbot.interview_dataset import InterviewDataset


class InterviewDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = InterviewDataset()

    def test_covers_all_requested_subjects_and_roles(self):
        expected_subjects = {
            "Programming", "Data Structures", "Algorithms", "OOP", "DBMS",
            "Operating Systems", "Computer Networks", "Software Engineering",
            "Computer Architecture", "Web Development", "Artificial Intelligence",
            "Machine Learning", "Deep Learning", "NLP", "Generative AI", "LLM",
            "Computer Vision", "Cybersecurity", "Cloud Computing", "DevOps",
            "System Design", "Distributed Systems", "Blockchain", "Data Science",
            "Software Testing", "Git and GitHub", "API and Web Services", "IoT",
            "Theory of Computation", "Compiler Design",
        }
        expected_roles = {
            "Software Developer", "Python Developer", "Java Developer",
            "Full Stack Developer", "Frontend Developer", "Backend Developer",
            "Data Analyst", "Data Scientist", "AI Engineer", "ML Engineer",
            "Generative AI Engineer", "Computer Vision Engineer", "DevOps Engineer",
            "Cloud Engineer", "Cybersecurity Engineer", "QA Engineer",
            "Blockchain Developer",
        }
        self.assertEqual(set(self.dataset.subjects()), expected_subjects)
        self.assertEqual(set(self.dataset.job_roles()), expected_roles)

    def test_generates_by_role_skill_subject_and_difficulty(self):
        questions = self.dataset.generate_questions(
            job_role="Python Developer",
            skills=["algorithms"],
            subject="Algorithms",
            difficulty="advanced",
            count=3,
        )
        self.assertEqual(len(questions), 3)
        self.assertTrue(all(question["difficulty"] == "advanced" for question in questions))
        self.assertTrue(all(question["subject"] == "Algorithms" for question in questions))
        self.assertTrue(all("Python Developer" in question["job_roles"] for question in questions))
        self.assertTrue(all("reference_answer" not in question for question in questions))

    def test_comparison_reference_covers_both_options_and_tradeoffs(self):
        question = next(
            record for record in self.dataset.records
            if record["subject"] == "Computer Networks"
            and record["question_type"] == "comparison"
            and record["difficulty"] == "intermediate"
        )
        answer = question["reference_answer"]
        for expected in ("Option A", "Option B", "Difference:", "Advantages:", "Disadvantages:", "Use when:", "Which is better:"):
            self.assertIn(expected, answer)

    def test_evaluates_answer_and_lists_missing_points(self):
        question_id = "algorithms_definition_easy"
        result = self.dataset.evaluate_answer(
            question_id,
            "Binary search requires sorted data and repeatedly halves the interval. It takes O(log n).",
        )
        self.assertGreaterEqual(result["score"], 5)
        self.assertLess(result["score"], 10)
        self.assertTrue(result["missing_points"])
        self.assertIn("improved_answer", result)
        self.assertIn("feedback", result)

    def test_short_binary_search_definition_scores_nine_and_suggests_middle_element(self):
        result = self.dataset.evaluate_answer(
            "algorithms_definition_easy",
            "Binary Search is a searching algorithm used to find an element in a sorted array "
            "by repeatedly dividing the search range into two halves.",
        )
        self.assertEqual(result["score"], 9.0)
        self.assertTrue(result["correct"])
        self.assertEqual(len(result["strengths"]), 3)
        self.assertIn("mention comparing the target with the middle element", result["improvement"])
        self.assertIn("middle element", result["improved_answer"])

    def test_semantic_evaluation_accepts_paraphrase_and_different_sentence_order(self):
        result = self.dataset.evaluate_answer(
            "algorithms_definition_easy",
            "The desired value can be found by repeatedly shrinking the candidate range. "
            "This method only applies when the values are already in ascending order.",
        )
        self.assertEqual(result["score"], 9.0)
        self.assertEqual(result["relevance"], "relevant")
        self.assertEqual(len(result["strengths"]), 3)
        self.assertIn("middle element", result["improvement"])

    def test_semantic_evaluation_flags_contradicted_precondition(self):
        result = self.dataset.evaluate_answer(
            "algorithms_definition_easy",
            "Binary search finds a target by halving the range and does not require sorted data.",
        )
        self.assertTrue(result["mistakes"])
        self.assertFalse(result["correct"])
        self.assertIn("sorted data", result["mistakes"][0]["correction"])

    def test_semantic_evaluation_marks_unrelated_answer_off_topic(self):
        result = self.dataset.evaluate_answer(
            "algorithms_definition_easy",
            "Photosynthesis converts light into chemical energy in plants.",
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["relevance"], "off-topic")

    def test_ordered_sequence_and_sequential_order_are_equivalent(self):
        question_id = "python_lists_easy"
        answer = self.dataset.evaluate_answer(
            question_id,
            "A Python list is mutable, and its items stay in sequential order. It stores several values.",
        )
        self.assertIn("ordered sequence", answer["matched_points"])
        self.assertEqual(answer["relevance"], "relevant")

    def test_rejects_unknown_filters_and_empty_answers(self):
        with self.assertRaises(ValueError):
            self.dataset.generate_questions(job_role="Unknown Role")
        with self.assertRaises(ValueError):
            self.dataset.evaluate_answer("algorithms_definition_easy", "  ")


if __name__ == "__main__":
    unittest.main()
