"""Offline tests for the minimal Spark-X2.6-4B health chatbot demo.

These tests never touch the network: the HTTP call is replaced with a stub.
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"),
)

import health_chatbot as chatbot


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = chatbot.build_messages("How do I look after a fever?")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], chatbot.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "How do I look after a fever?")

    def test_question_is_stripped(self):
        messages = chatbot.build_messages("   hello   ")
        self.assertEqual(messages[1]["content"], "hello")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            chatbot.build_messages("   ")
        with self.assertRaises(ValueError):
            chatbot.build_messages(None)

    def test_safety_rules_are_in_the_prompt(self):
        prompt = chatbot.SYSTEM_PROMPT.lower()
        for keyword in ("diagnose", "prescribe", "clinician", "emergency"):
            self.assertIn(keyword, prompt)


class BuildPayloadTest(unittest.TestCase):
    def test_uses_model_id_and_token_budget(self):
        payload = chatbot.build_payload("Is a cough normal?")
        self.assertEqual(payload["model"], chatbot.MODEL_ID)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], chatbot.MAX_NEW_TOKENS)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])


class GenerateTest(unittest.TestCase):
    def test_returns_the_model_answer(self):
        body = {"choices": [{"message": {"content": "  Rest and see a clinician.  "}}]}
        with mock.patch.object(
            chatbot.urllib.request, "urlopen", return_value=_FakeResponse(body)
        ):
            answer = chatbot.generate("test question")
        self.assertEqual(answer, "Rest and see a clinician.")

    def test_malformed_response_raises(self):
        with mock.patch.object(
            chatbot.urllib.request, "urlopen", return_value=_FakeResponse({"error": "x"})
        ):
            with self.assertRaises(ValueError):
                chatbot.generate("test question")

    def test_unreachable_endpoint_raises_runtime_error(self):
        with mock.patch.object(
            chatbot.urllib.request,
            "urlopen",
            side_effect=chatbot.urllib.error.URLError("boom"),
        ):
            with self.assertRaises(RuntimeError):
                chatbot.generate("test question")


if __name__ == "__main__":
    unittest.main()
