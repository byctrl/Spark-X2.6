"""Tests for the HER Hack-Astron #7 health-information chatbot demo.

Run from the repository root with:
    python -m unittest discover -s src
"""

import json
import unittest

import health_chatbot


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOpener:
    def __init__(self, payload):
        self._payload = payload
        self.request = None
        self.timeout = None

    def __call__(self, request, timeout=None):
        self.request = request
        self.timeout = timeout
        return _FakeResponse(self._payload)


class BuildMessagesTest(unittest.TestCase):
    def test_messages_put_system_prompt_first(self):
        messages = health_chatbot.build_messages("What is a fever?")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], health_chatbot.SYSTEM_PROMPT)
        self.assertEqual(messages[1], {"role": "user", "content": "What is a fever?"})

    def test_question_is_stripped(self):
        messages = health_chatbot.build_messages("  cough  ")
        self.assertEqual(messages[1]["content"], "cough")

    def test_empty_question_is_rejected(self):
        for value in ("", "   ", None):
            with self.assertRaises(ValueError):
                health_chatbot.build_messages(value)

    def test_system_prompt_keeps_safety_boundaries(self):
        prompt = health_chatbot.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("clinician", prompt)


class BuildPayloadTest(unittest.TestCase):
    def test_payload_uses_spark_model_and_token_budget(self):
        payload = health_chatbot.build_payload("Is this urgent?")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["model"], health_chatbot.MODEL_ID)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["max_tokens"], health_chatbot.MAX_NEW_TOKENS)
        self.assertEqual(payload["messages"][1]["content"], "Is this urgent?")

    def test_payload_keeps_demo_question_usable(self):
        payload = health_chatbot.build_payload(health_chatbot.DEMO_QUESTION)
        self.assertTrue(payload["messages"][1]["content"])


class ParseReplyTest(unittest.TestCase):
    def test_returns_trimmed_content(self):
        data = {"choices": [{"message": {"content": "  See a clinician.  "}}]}
        self.assertEqual(health_chatbot.parse_reply(data), "See a clinician.")

    def test_rejects_empty_content(self):
        data = {"choices": [{"message": {"content": "   "}}]}
        with self.assertRaises(ValueError):
            health_chatbot.parse_reply(data)

    def test_rejects_unexpected_shape(self):
        for data in ({}, {"choices": []}, {"choices": [{}]}, None):
            with self.assertRaises(ValueError):
                health_chatbot.parse_reply(data)


class AskTest(unittest.TestCase):
    def test_posts_to_chat_completions_and_returns_answer(self):
        opener = _FakeOpener(
            {"choices": [{"message": {"content": "Here is plain-language info."}}]}
        )
        answer = health_chatbot.ask(
            "What is a fever?", base_url="http://127.0.0.1:30000/v1", opener=opener
        )
        self.assertEqual(answer, "Here is plain-language info.")
        self.assertTrue(opener.request.full_url.endswith("/v1/chat/completions"))
        body = json.loads(opener.request.data.decode("utf-8"))
        self.assertEqual(body["model"], health_chatbot.MODEL_ID)
        self.assertEqual(body["messages"][1]["content"], "What is a fever?")

    def test_trailing_slash_in_base_url_is_handled(self):
        opener = _FakeOpener({"choices": [{"message": {"content": "ok"}}]})
        health_chatbot.ask("hi", base_url="http://localhost:8000/v1/", opener=opener)
        self.assertEqual(opener.request.full_url, "http://localhost:8000/v1/chat/completions")

    def test_model_error_is_not_silently_accepted(self):
        opener = _FakeOpener({"error": "model not loaded"})
        with self.assertRaises(ValueError):
            health_chatbot.ask("hi", opener=opener)


if __name__ == "__main__":
    unittest.main()
