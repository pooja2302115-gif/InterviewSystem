import unittest

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.chatbot.conversation import ConversationManager


class FakeGenerator:
    def generate(self, prompt):
        return "Practice response."


class ApiTests(unittest.TestCase):
    def make_client(self):
        manager = ConversationManager(FakeGenerator())
        return TestClient(create_app(manager=manager))

    def test_health_reports_loaded_model(self):
        response = self.make_client().get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model_loaded": True})

    def test_chat_returns_session_id_and_history(self):
        client = self.make_client()
        first = client.post("/chat", json={"message": "What is a stack?"})
        session_id = first.json()["session_id"]
        second = client.post(
            "/chat",
            json={"message": "What is its complexity?", "session_id": session_id},
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["session_id"], session_id)
        history = client.get(f"/sessions/{session_id}/messages")
        self.assertEqual(len(history.json()["messages"]), 4)

    def test_invalid_message_is_rejected(self):
        response = self.make_client().post("/chat", json={"message": ""})
        self.assertEqual(response.status_code, 422)

    def test_unconfigured_app_returns_service_unavailable(self):
        client = TestClient(create_app())
        self.assertEqual(client.get("/health").json()["model_loaded"], False)
        response = client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 503)

    def test_interview_question_generation_and_answer_evaluation(self):
        client = self.make_client()
        generated = client.post(
            "/interview/questions",
            json={
                "job_role": "Python Developer",
                "skills": ["algorithms"],
                "subject": "Algorithms",
                "difficulty": "easy",
                "count": 1,
            },
        )
        self.assertEqual(generated.status_code, 200)
        question = generated.json()["questions"][0]
        self.assertEqual(question["subject"], "Algorithms")
        self.assertEqual(question["difficulty"], "easy")
        self.assertNotIn("reference_answer", question)

        evaluated = client.post(
            "/interview/evaluate",
            json={
                "question_id": question["id"],
                "answer": "Binary search needs sorted data and halves the search interval in O(log n).",
            },
        )
        self.assertEqual(evaluated.status_code, 200)
        self.assertIn("score", evaluated.json())
        self.assertIn("mistakes", evaluated.json())
        self.assertIn("missing_points", evaluated.json())
        self.assertIn("improved_answer", evaluated.json())

    def test_interview_api_rejects_invalid_job_role(self):
        response = self.make_client().post(
            "/interview/questions",
            json={"job_role": "Unknown Role"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
