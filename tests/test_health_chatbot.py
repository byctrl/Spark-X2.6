"""Unit tests for src/health_chatbot.py (no network, no model required)."""

import json
import os
import sys
import unittest
from unittest import mock

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import health_chatbot as hc  # noqa: E402


class FakeResponse(object):
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def completion(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class BuildMessagesTests(unittest.TestCase):
    def test_system_prompt_and_question(self):
        messages = hc.build_messages("Apa itu demam?")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "Apa itu demam?"})

    def test_rejects_blank_question(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_keeps_valid_history_only(self):
        history = [
            {"role": "user", "content": "Halo"},
            {"role": "assistant", "content": "Halo, ada yang bisa dibantu?"},
            {"role": "tool", "content": "ignored"},
            {"role": "user", "content": ""},
        ]
        messages = hc.build_messages("Pertanyaan", history)
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])


class PayloadTests(unittest.TestCase):
    def test_uses_spark_model_id(self):
        payload = hc.build_payload(hc.build_messages("Halo"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])


class ExtractAnswerTests(unittest.TestCase):
    def test_appends_disclaimer(self):
        answer = hc.extract_answer(completion("Minum air cukup dan istirahat."))
        self.assertTrue(answer.startswith("Minum air cukup dan istirahat."))
        self.assertIn(hc.DISCLAIMER, answer)

    def test_does_not_duplicate_disclaimer(self):
        text = "Istirahat cukup.\n\n" + hc.DISCLAIMER
        self.assertEqual(hc.extract_answer(completion(text)), text)

    def test_rejects_malformed_or_empty(self):
        with self.assertRaises(ValueError):
            hc.extract_answer({})
        with self.assertRaises(ValueError):
            hc.extract_answer(completion("   "))


class AskTests(unittest.TestCase):
    def test_posts_to_chat_completions(self):
        fake = mock.patch.object(
            hc.urllib.request,
            "urlopen",
            return_value=FakeResponse(completion("Jawaban singkat.")),
        )
        with fake as urlopen:
            answer = hc.ask("Apa itu demam?", base_url="http://localhost:9999/v1/")

        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://localhost:9999/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], hc.MODEL_ID)
        self.assertEqual(body["messages"][-1]["content"], "Apa itu demam?")
        self.assertIn(hc.DISCLAIMER, answer)


if __name__ == "__main__":
    unittest.main()
