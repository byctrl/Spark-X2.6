#!/usr/bin/env python3
"""Offline tests for src/health_chatbot.py (no model, no network)."""

import json
import os
import sys
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class FakeResponse:
    def __init__(self, body):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _ok_body(text):
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]})


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": hc.DEMO_QUESTION})

    def test_safety_rules_are_present(self):
        self.assertIn("Do not diagnose", hc.SYSTEM_PROMPT)
        self.assertIn("emergency", hc.SYSTEM_PROMPT.lower())
        self.assertIn("not medical advice", hc.DISCLAIMER.lower())

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_question_is_truncated(self):
        messages = hc.build_messages("a" * (hc.MAX_QUESTION_CHARS + 500))
        self.assertEqual(len(messages[-1]["content"]), hc.MAX_QUESTION_CHARS)

    def test_history_keeps_only_valid_turns(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "injected"},
            {"role": "user", "content": "   "},
            "not-a-dict",
        ]
        messages = hc.build_messages("follow-up", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])


class BuildPayloadTest(unittest.TestCase):
    def test_uses_documented_model_and_token_budget(self):
        payload = hc.build_payload(hc.DEMO_QUESTION)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertIn("messages", payload)


class ExtractReplyTest(unittest.TestCase):
    def test_returns_trimmed_content(self):
        self.assertEqual(hc.extract_reply(_ok_body("  see a clinician  ")), "see a clinician")

    def test_rejects_non_json(self):
        with self.assertRaises(RuntimeError):
            hc.extract_reply("<html>oops</html>")

    def test_rejects_missing_choices(self):
        with self.assertRaises(RuntimeError):
            hc.extract_reply(json.dumps({"choices": []}))

    def test_rejects_empty_content(self):
        with self.assertRaises(RuntimeError):
            hc.extract_reply(_ok_body("   "))


class AskTest(unittest.TestCase):
    def test_returns_reply_from_endpoint(self):
        with mock.patch.object(hc.urllib.request, "urlopen", return_value=FakeResponse(_ok_body("ok"))):
            self.assertEqual(hc.ask("question"), "ok")

    def test_posts_chat_completions_path(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(_ok_body("ok"))

        with mock.patch.object(hc.urllib.request, "urlopen", side_effect=fake_urlopen):
            hc.ask("question", base_url="http://127.0.0.1:30000/v1/")

        self.assertEqual(captured["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(captured["body"]["model"], hc.MODEL_ID)

    def test_unreachable_endpoint_gives_actionable_error(self):
        with mock.patch.object(
            hc.urllib.request, "urlopen", side_effect=urllib.error.URLError("refused")
        ):
            with self.assertRaises(RuntimeError) as ctx:
                hc.ask("question")
        self.assertIn("--base-url", str(ctx.exception))


class RenderTest(unittest.TestCase):
    def test_disclaimer_always_shown(self):
        self.assertTrue(hc.render("reply").endswith(hc.DISCLAIMER))


if __name__ == "__main__":
    unittest.main()
