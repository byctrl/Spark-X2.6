#!/usr/bin/env python3
"""Offline tests for src/health_chatbot.py (no network, no extra dependencies).

Run from the repository root:  python src/test_health_chatbot.py
"""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})
        self.assertEqual(len(messages), 2)

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
        ]
        messages = hc.build_messages("What is a fever?", history)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])

    def test_empty_question_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_bad_history_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("q", [{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("q", [{"role": "user", "content": "  "}])

    def test_system_prompt_has_safety_boundaries(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("clinician", prompt)


class PayloadAndParseTest(unittest.TestCase):
    def test_payload_uses_spark_model(self):
        payload = hc.build_payload(hc.build_messages("q"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])

    def test_parse_response_strips_content(self):
        data = {"choices": [{"message": {"content": "  hello  "}}]}
        self.assertEqual(hc.parse_response(data), "hello")

    def test_parse_response_rejects_empty(self):
        with self.assertRaises(hc.ModelResponseError):
            hc.parse_response({"choices": [{"message": {"content": "  "}}]})
        with self.assertRaises(hc.ModelResponseError):
            hc.parse_response({})


class AnswerTest(unittest.TestCase):
    def test_answer_uses_injected_transport(self):
        seen = {}

        def fake_poster(url, payload, timeout):
            seen["url"] = url
            seen["payload"] = payload
            return {"choices": [{"message": {"content": "A fever is a higher body temperature."}}]}

        reply = hc.answer("What is a fever?", base_url="http://127.0.0.1:30000/v1/", poster=fake_poster)
        self.assertEqual(seen["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(seen["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(reply["model"], hc.MODEL_ID)
        self.assertEqual(reply["answer"], "A fever is a higher body temperature.")
        self.assertEqual(reply["notice"], hc.SAFETY_FOOTER)
        self.assertIn("is general health education", hc.render(reply))


class CliTest(unittest.TestCase):
    def test_offline_prints_payload_without_network(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = hc.main(["--offline"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["messages"][-1]["content"], hc.DEMO_QUESTION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
