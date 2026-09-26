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


if __name__ == "__main__":
    unittest.main()
