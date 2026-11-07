"""Offline tests for src/health_chatbot.py (no model and no network needed)."""

import json
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import health_chatbot as hc  # noqa: E402


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
    def test_system_prompt_then_question(self):
        messages = hc.build_messages("  What is hydration?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is hydration?"})
        self.assertEqual(len(messages), 2)

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        messages = hc.build_messages("next", history)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])

    def test_empty_question_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_bad_history_role_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("q", [{"role": "system", "content": "x"}])

    def test_prompt_forbids_diagnosis(self):
        self.assertIn("diagnose", hc.SYSTEM_PROMPT.lower())
        self.assertIn("clinician", hc.SYSTEM_PROMPT.lower())


class BuildPayloadTest(unittest.TestCase):
    def test_defaults(self):
        payload = hc.build_payload(hc.build_messages("q"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])

    def test_empty_messages_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_payload([])

    def test_invalid_max_tokens_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_payload(hc.build_messages("q"), max_tokens=0)


class ParseResponseTest(unittest.TestCase):
    def test_extracts_and_strips_content(self):
        data = {"choices": [{"message": {"role": "assistant", "content": "  hi  "}}]}
        self.assertEqual(hc.parse_response(data), "hi")

    def test_missing_choices_rejected(self):
        for bad in ({}, {"choices": []}, None):
            with self.assertRaises(ValueError):
                hc.parse_response(bad)

    def test_blank_content_rejected(self):
        data = {"choices": [{"message": {"content": "   "}}]}
        with self.assertRaises(ValueError):
            hc.parse_response(data)


class AskTest(unittest.TestCase):
    def test_sends_expected_request(self):
        captured = {}

        def fake_opener(request, timeout=None):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["auth"] = request.get_header("Authorization")
            captured["timeout"] = timeout
            payload = {"choices": [{"message": {"content": "drink water"}}]}
            return _FakeResponse(payload)

        reply = hc.ask(
            "demo",
            base_url="http://127.0.0.1:30000/v1/",
            api_key="secret",
            model="XHToken/Spark-X2.6-4B",
            opener=fake_opener,
        )

        self.assertEqual(reply, "drink water")
        self.assertEqual(captured["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(captured["auth"], "Bearer secret")
        self.assertEqual(captured["body"]["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(captured["body"]["messages"][0]["role"], "system")
        self.assertEqual(captured["body"]["messages"][-1]["content"], "demo")


if __name__ == "__main__":
    unittest.main()
