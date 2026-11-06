"""Unit tests for the minimal Spark-X2.6-4B health chatbot demo."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import health_chatbot as hc  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _opener(payload, captured=None):
    def _open(request, timeout=None):
        if captured is not None:
            captured.append(request)
        return _FakeResponse(payload)

    return _open


class BuildMessagesTests(unittest.TestCase):
    def test_system_prompt_comes_first_and_question_last(self):
        messages = hc.build_messages("  Is a mild fever normal?  ")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "Is a mild fever normal?"})

    def test_history_turns_are_kept_in_order(self):
        history = [
            {"role": "user", "content": "What is a fever?"},
            {"role": "assistant", "content": "A raised body temperature."},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("And now?", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class PayloadTests(unittest.TestCase):
    def test_payload_targets_spark_model_with_bounded_output(self):
        payload = hc.build_payload(hc.build_messages(hc.DEMO_QUESTION))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertTrue(payload["messages"])

    def test_empty_messages_are_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_payload([])

    def test_url_join_is_stable(self):
        self.assertEqual(
            hc.chat_completions_url("http://127.0.0.1:30000/v1/"),
            "http://127.0.0.1:30000/v1/chat/completions",
        )


class ResponseTests(unittest.TestCase):
    def test_reply_gets_disclaimer_appended_once(self):
        reply = hc.format_reply("Rest and drink water.")
        self.assertIn("Rest and drink water.", reply)
        self.assertIn(hc.DISCLAIMER, reply)
        self.assertEqual(hc.format_reply(reply), reply)

    def test_malformed_response_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.parse_response({"choices": []})

    def test_empty_answer_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.parse_response({"choices": [{"message": {"content": "  "}}]})

    def test_ask_posts_question_and_returns_answer(self):
        captured = []
        body = {"choices": [{"message": {"content": "See a clinician if it lasts."}}]}
        answer = hc.ask(hc.DEMO_QUESTION, opener=_opener(body, captured))
        self.assertIn("See a clinician if it lasts.", answer)
        self.assertIn(hc.DISCLAIMER, answer)
        self.assertEqual(len(captured), 1)
        sent = json.loads(captured[0].data.decode("utf-8"))
        self.assertEqual(sent["model"], hc.MODEL_ID)
        self.assertEqual(sent["messages"][-1]["content"], hc.DEMO_QUESTION)

    def test_ask_reports_unreachable_endpoint(self):
        import urllib.error

        def _boom(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        with self.assertRaises(RuntimeError):
            hc.ask("hello", opener=_boom)


if __name__ == "__main__":
    unittest.main()
