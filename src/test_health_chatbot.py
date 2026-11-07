"""Tests for the minimal Spark-X2.6-4B health chatbot demo."""

import json
import unittest

import health_chatbot


class _FakeResponse(object):
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = health_chatbot.build_messages("hello")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], health_chatbot.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "hello"})

    def test_history_is_filtered_to_user_and_assistant(self):
        history = [
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "tool", "content": "ignore me too"},
        ]
        messages = health_chatbot.build_messages("new question", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_system_prompt_keeps_safety_boundaries(self):
        prompt = health_chatbot.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("qualified health worker", prompt)
        self.assertIn("emergency", prompt)


class PayloadTest(unittest.TestCase):
    def test_payload_uses_model_and_token_limit(self):
        payload = health_chatbot.build_payload(health_chatbot.DEMO_QUESTION)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], health_chatbot.MAX_NEW_TOKENS)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["messages"][-1]["content"], health_chatbot.DEMO_QUESTION)

    def test_demo_question_is_not_empty(self):
        self.assertTrue(health_chatbot.DEMO_QUESTION.strip())


class ParseCompletionTest(unittest.TestCase):
    def test_valid_payload(self):
        payload = {"choices": [{"message": {"content": "  drink water  "}}]}
        self.assertEqual(health_chatbot.parse_completion(payload), "drink water")

    def test_invalid_payload_raises(self):
        with self.assertRaises(ValueError):
            health_chatbot.parse_completion({"choices": []})

    def test_empty_content_raises(self):
        payload = {"choices": [{"message": {"content": "   "}}]}
        with self.assertRaises(ValueError):
            health_chatbot.parse_completion(payload)


class AskTest(unittest.TestCase):
    def test_ask_uses_injected_opener(self):
        captured = {}

        def opener(request, timeout=None):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _FakeResponse({"choices": [{"message": {"content": "see a clinician"}}]})

        answer = health_chatbot.ask("cough", base_url="http://127.0.0.1:9/v1/", opener=opener)
        self.assertEqual(answer, "see a clinician")
        self.assertEqual(captured["url"], "http://127.0.0.1:9/v1/chat/completions")
        self.assertEqual(captured["body"]["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(captured["timeout"], health_chatbot.TIMEOUT_SECONDS)


if __name__ == "__main__":
    unittest.main()
