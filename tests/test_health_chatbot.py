"""Offline tests for src/health_chatbot.py (no model, no network)."""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc  # noqa: E402


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _ok_body(text):
    return json.dumps({"choices": [{"message": {"content": text}}]}).encode("utf-8")


class ConstantsTest(unittest.TestCase):
    def test_model_id_is_spark_x26_4b(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_forbids_diagnosis_and_prescription(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("never prescribe", prompt)
        self.assertIn("emergency", prompt)

    def test_demo_question_exists(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  Apa itu demam?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Apa itu demam?"})

    def test_history_is_preserved_in_order(self):
        history = [
            {"role": "user", "content": "Saya batuk."},
            {"role": "assistant", "content": "Sejak kapan?"},
        ]
        messages = hc.build_messages("Sudah tiga hari.", history)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None, 5):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_invalid_history_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hi", [{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("hi", [{"role": "user", "content": ""}])
        with self.assertRaises(ValueError):
            hc.build_messages("hi", ["not-a-dict"])


class PayloadTest(unittest.TestCase):
    def test_payload_uses_model_and_token_budget(self):
        messages = hc.build_messages("hello")
        payload = hc.build_payload(messages)
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(payload["messages"], messages)
        self.assertGreater(len(payload["messages"]), 0)

    def test_empty_messages_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_payload([])

    def test_url_is_openai_compatible(self):
        self.assertEqual(hc.chat_completions_url("http://127.0.0.1:30000/v1/"),
                         "http://127.0.0.1:30000/v1/chat/completions")


class ParseResponseTest(unittest.TestCase):
    def test_extracts_and_strips_content(self):
        data = {"choices": [{"message": {"content": "  Minum air.  "}}]}
        self.assertEqual(hc.parse_response(data), "Minum air.")

    def test_rejects_malformed_response(self):
        for bad in ({}, {"choices": []}, {"choices": [{}]}, None):
            with self.assertRaises(ValueError):
                hc.parse_response(bad)

    def test_rejects_empty_answer(self):
        for text in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.parse_response({"choices": [{"message": {"content": text}}]})


class AskTest(unittest.TestCase):
    def test_ask_posts_to_endpoint_and_returns_answer(self):
        with mock.patch.object(hc.urllib.request, "urlopen",
                               return_value=_FakeResponse(_ok_body("Ke klinik ya."))) as opener:
            answer = hc.ask("Saya demam.", base_url="http://localhost:9999/v1")

        self.assertEqual(answer, "Ke klinik ya.")
        request = opener.call_args[0][0]
        self.assertEqual(request.full_url, "http://localhost:9999/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["model"], hc.MODEL_ID)
        self.assertEqual(sent["messages"][-1]["content"], "Saya demam.")

    def test_ask_reports_unreachable_endpoint(self):
        error = hc.urllib.error.URLError("connection refused")
        with mock.patch.object(hc.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as ctx:
                hc.ask("hi", base_url="http://localhost:9999/v1")
        self.assertIn("SPARK_BASE_URL", str(ctx.exception))

    def test_ask_rejects_empty_model_answer(self):
        with mock.patch.object(hc.urllib.request, "urlopen",
                               return_value=_FakeResponse(_ok_body("  "))):
            with self.assertRaises(ValueError):
                hc.ask("hi")


class RenderTest(unittest.TestCase):
    def test_render_includes_safety_note(self):
        out = hc.render("Jawaban.")
        self.assertIn("Jawaban.", out)
        self.assertIn(hc.SAFETY_NOTE, out)


if __name__ == "__main__":
    unittest.main()
