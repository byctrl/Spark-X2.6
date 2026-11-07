"""Tests for the minimal Spark-X2.6-4B health chatbot demo."""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import health_chatbot  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_is_first_and_question_is_last(self):
        messages = health_chatbot.build_messages("  what is a fever?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], health_chatbot.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "what is a fever?"})

    def test_system_prompt_states_safety_boundary(self):
        prompt = health_chatbot.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("not medical advice", prompt)

    def test_blank_question_is_rejected(self):
        for value in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                health_chatbot.build_messages(value)

    def test_question_is_truncated(self):
        messages = health_chatbot.build_messages("a" * 50, max_chars=10)
        self.assertEqual(messages[-1]["content"], "a" * 10)

    def test_history_is_kept_between_system_and_question(self):
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        messages = health_chatbot.build_messages("more", history=history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_bad_history_is_rejected(self):
        with self.assertRaises(ValueError):
            health_chatbot.build_messages("q", history=[{"role": "system", "content": "x"}])


class PayloadTest(unittest.TestCase):
    def test_payload_targets_spark_and_bounds_output(self):
        payload = health_chatbot.build_payload(health_chatbot.build_messages("q"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], health_chatbot.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])

    def test_empty_messages_are_rejected(self):
        with self.assertRaises(ValueError):
            health_chatbot.build_payload([])


class ParseCompletionTest(unittest.TestCase):
    def test_extracts_and_strips_answer(self):
        data = {"choices": [{"message": {"content": "  See a clinician.  "}}]}
        self.assertEqual(health_chatbot.parse_completion(data), "See a clinician.")

    def test_rejects_malformed_or_empty_answer(self):
        for data in ({}, {"choices": []}, {"choices": [{"message": {"content": "  "}}]}):
            with self.assertRaises(ValueError):
                health_chatbot.parse_completion(data)


class AskTest(unittest.TestCase):
    def test_posts_to_chat_completions_and_returns_answer(self):
        payload = {"choices": [{"message": {"content": "Ask the clinic about travel."}}]}
        with mock.patch.object(
            health_chatbot.urllib.request, "urlopen", return_value=_FakeResponse(payload)
        ) as fake:
            answer = health_chatbot.ask("cough", base_url="http://localhost:9/v1/")

        self.assertEqual(answer, "Ask the clinic about travel.")
        request = fake.call_args[0][0]
        self.assertEqual(request.full_url, "http://localhost:9/v1/chat/completions")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], health_chatbot.MODEL_ID)


if __name__ == "__main__":
    unittest.main()
