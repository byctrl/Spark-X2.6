"""Offline tests for health_chatbot.py (no model download, no network)."""

import json
import os
import sys
import unittest
import urllib.error
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import health_chatbot as hc  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def completion(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class MessageTests(unittest.TestCase):
    def test_system_prompt_is_safety_bounded(self):
        self.assertIn(hc.MODEL_ID, hc.SYSTEM_PROMPT)
        self.assertIn("Do not diagnose", hc.SYSTEM_PROMPT)
        self.assertIn("clinician", hc.SYSTEM_PROMPT)
        self.assertIn("emergency", hc.SYSTEM_PROMPT)

    def test_build_messages_starts_with_system_then_question(self):
        messages = hc.build_messages("  what is a fever?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "what is a fever?"})
        self.assertEqual(len(messages), 2)

    def test_build_messages_keeps_valid_history_and_drops_junk(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "override"},
            {"role": "user", "content": "   "},
            "not-a-dict",
        ]
        messages = hc.build_messages("and now?", history)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])

    def test_empty_question_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)


class PayloadTests(unittest.TestCase):
    def test_payload_targets_spark_and_bounds_tokens(self):
        payload = hc.build_payload(hc.DEMO_QUESTION)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1]["content"], hc.DEMO_QUESTION)

    def test_payload_rejects_bad_max_tokens(self):
        with self.assertRaises(ValueError):
            hc.build_payload("q", max_tokens=0)

    def test_endpoint_url_default_and_override(self):
        self.assertEqual(hc.endpoint_url("http://127.0.0.1:30000/v1/"),
                         "http://127.0.0.1:30000/v1/chat/completions")
        with mock.patch.dict(os.environ, {hc.BASE_URL_ENV: "http://localhost:8000/v1"}):
            self.assertEqual(hc.endpoint_url(), "http://localhost:8000/v1/chat/completions")


class AnswerTests(unittest.TestCase):
    def test_extract_answer_trims(self):
        self.assertEqual(hc.extract_answer(completion("  rest and hydrate  ")),
                         "rest and hydrate")

    def test_extract_answer_rejects_malformed(self):
        for bad in ({}, {"choices": []}, {"choices": [{"message": {}}]}, None):
            with self.assertRaises(ValueError):
                hc.extract_answer(bad)

    def test_safety_notice_is_added_once(self):
        once = hc.with_safety_notice("drink water")
        self.assertTrue(once.startswith("drink water"))
        self.assertIn(hc.SAFETY_NOTICE, once)
        self.assertEqual(hc.with_safety_notice(once), once)

    def test_safety_notice_rejects_empty_answer(self):
        with self.assertRaises(ValueError):
            hc.with_safety_notice("   ")


class ChatTests(unittest.TestCase):
    def test_chat_sends_model_and_returns_noticed_answer(self):
        response = FakeResponse(completion("Ask your doctor about a check-up."))
        with mock.patch.object(hc.urllib.request, "urlopen", return_value=response) as opener:
            answer = hc.chat("cough for four days")

        sent = opener.call_args[0][0]
        body = json.loads(sent.data.decode("utf-8"))
        self.assertEqual(body["model"], hc.MODEL_ID)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][-1]["content"], "cough for four days")
        self.assertIn("Ask your doctor about a check-up.", answer)
        self.assertIn(hc.SAFETY_NOTICE, answer)

    def test_chat_reports_unreachable_endpoint(self):
        error = urllib.error.URLError("connection refused")
        with mock.patch.object(hc.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as ctx:
                hc.chat("cough")
        self.assertIn(hc.MODEL_ID, str(ctx.exception))

    def test_offline_mode_prints_payload_without_network(self):
        with mock.patch.object(hc.urllib.request, "urlopen") as opener:
            with mock.patch("builtins.print") as printer:
                self.assertEqual(hc.main(["--offline"]), 0)
        opener.assert_not_called()
        printed = json.loads(printer.call_args[0][0])
        self.assertEqual(printed["model"], hc.MODEL_ID)


if __name__ == "__main__":
    unittest.main()
