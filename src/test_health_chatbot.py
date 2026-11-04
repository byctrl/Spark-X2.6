"""Offline unit tests for src/health_chatbot.py.

These tests never touch the network; the HTTP call is injected.
Run from the repository root with:

    python -m unittest discover -s src -p "test_*.py" -v
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc  # noqa: E402


class _FakeResponse:
    def __init__(self, body):
        self._body = body
        self.closed = False

    def read(self):
        return self._body

    def close(self):
        self.closed = True


class HealthChatbotTests(unittest.TestCase):
    def test_messages_start_with_system_prompt(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_preserved_in_order(self):
        history = [{"role": "assistant", "content": "Hello"}]
        messages = hc.build_messages("And then?", history)
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "assistant", "user"],
        )

    def test_payload_uses_the_published_model(self):
        payload = hc.build_payload(hc.DEMO_QUESTION)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)

    def test_system_prompt_keeps_safety_boundaries(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("emergency", prompt)

    def test_chat_posts_to_the_configured_endpoint(self):
        captured = {}

        def opener(request, timeout=None):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            body = json.dumps({"choices": [{"message": {"content": "  Short answer.  "}}]})
            return _FakeResponse(body.encode("utf-8"))

        answer = hc.chat("What is a fever?", base_url="http://localhost:9999/v1/", opener=opener)

        self.assertEqual(answer, "Short answer.")
        self.assertEqual(captured["url"], "http://localhost:9999/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(captured["timeout"], hc.REQUEST_TIMEOUT)

    def test_response_is_closed_after_reading(self):
        response = _FakeResponse(b"{\"choices\": [{\"message\": {\"content\": \"ok\"}}]}")

        def opener(request, timeout=None):
            return response

        self.assertEqual(hc.chat("Q", base_url="http://localhost:9999/v1", opener=opener), "ok")
        self.assertTrue(response.closed)

    def test_malformed_response_is_rejected(self):
        def opener(request, timeout=None):
            return _FakeResponse(b"{}")

        with self.assertRaises(ValueError):
            hc.chat("Q", base_url="http://localhost:9999/v1", opener=opener)

    def test_empty_answer_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.extract_text({"choices": [{"message": {"content": "   "}}]})


if __name__ == "__main__":
    unittest.main()
