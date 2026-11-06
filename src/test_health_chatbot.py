"""Offline tests for the Spark-X2.6-4B health chatbot demo."""

import json
import unittest

import health_chatbot as hc


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class HealthChatbotTest(unittest.TestCase):
    def test_uses_spark_model(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_token_budget_is_bounded(self):
        self.assertGreater(hc.MAX_NEW_TOKENS, 0)
        self.assertLessEqual(hc.MAX_NEW_TOKENS, 4096)

    def test_system_prompt_has_safety_boundaries(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("emergency", prompt)
        self.assertIn("not medical advice", prompt)

    def test_build_messages_starts_with_system_prompt(self):
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], hc.DEMO_QUESTION)

    def test_build_messages_keeps_valid_history(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "   "},
            "not a dict",
        ]
        messages = hc.build_messages("next question", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_empty_question_rejected(self):
        for bad in ("", "   ", None, 5):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_build_payload_fields(self):
        payload = hc.build_payload(hc.build_messages("q"), max_new_tokens=64)
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], 64)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "system")

    def test_extract_text(self):
        body = {"choices": [{"message": {"role": "assistant", "content": "  answer  "}}]}
        self.assertEqual(hc.extract_text(body), "answer")

    def test_extract_text_rejects_empty_response(self):
        with self.assertRaises(RuntimeError):
            hc.extract_text({"choices": []})
        with self.assertRaises(RuntimeError):
            hc.extract_text({"choices": [{"message": {"content": "  "}}]})

    def test_complete_uses_local_endpoint(self):
        seen = {}

        def fake_opener(request, timeout=None):
            seen["url"] = request.full_url
            seen["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse({"choices": [{"message": {"content": "stay hydrated"}}]})

        answer = hc.complete(
            hc.build_messages(hc.DEMO_QUESTION),
            base_url="http://127.0.0.1:30000/v1",
            opener=fake_opener,
        )
        self.assertEqual(answer, "stay hydrated")
        self.assertEqual(seen["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(seen["body"]["model"], hc.MODEL_ID)


if __name__ == "__main__":
    unittest.main()
