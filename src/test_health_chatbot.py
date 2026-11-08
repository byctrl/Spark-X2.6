"""Offline tests for the minimal Spark-X2.6-4B health chatbot."""

import json
import unittest

import health_chatbot


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class HealthChatbotTest(unittest.TestCase):
    def test_model_and_budget(self):
        self.assertEqual(health_chatbot.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(health_chatbot.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_is_safety_bounded(self):
        prompt = health_chatbot.SYSTEM_PROMPT.lower()
        for phrase in ("do not diagnose", "emergency", "not medical advice"):
            self.assertIn(phrase, prompt)

    def test_demo_question_present(self):
        self.assertTrue(health_chatbot.DEMO_QUESTION.strip())

    def test_build_messages(self):
        messages = health_chatbot.build_messages("  What is a fever?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], health_chatbot.SYSTEM_PROMPT)
        self.assertEqual(
            messages[1], {"role": "user", "content": "What is a fever?"}
        )

    def test_build_messages_rejects_empty(self):
        with self.assertRaises(ValueError):
            health_chatbot.build_messages("   ")

    def test_build_payload_uses_model(self):
        payload = health_chatbot.build_payload("hello")
        self.assertEqual(payload["model"], health_chatbot.MODEL_ID)
        self.assertEqual(payload["max_tokens"], health_chatbot.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])

    def test_chat_returns_reply(self):
        captured = {}

        def fake_opener(request, timeout=None):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {"choices": [{"message": {"content": "  A fever is ...  "}}]}
            )

        reply = health_chatbot.chat(
            "What is a fever?",
            base_url="http://localhost:30000/v1/",
            opener=fake_opener,
        )
        self.assertEqual(reply, "A fever is ...")
        self.assertEqual(
            captured["url"], "http://localhost:30000/v1/chat/completions"
        )
        self.assertEqual(captured["body"]["model"], health_chatbot.MODEL_ID)

    def test_chat_rejects_malformed_response(self):
        def broken_opener(request, timeout=None):
            return FakeResponse({"unexpected": True})

        with self.assertRaises(RuntimeError):
            health_chatbot.chat("What is a fever?", opener=broken_opener)


if __name__ == "__main__":
    unittest.main()
