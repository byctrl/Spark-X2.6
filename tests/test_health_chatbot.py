"""Tests for the minimal Spark-X2.6 health-information chatbot."""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import health_chatbot as hc  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _ok_payload(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_carries_safety_rules(self):
        messages = hc.build_messages("I have a fever.")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        system = messages[0]["content"]
        self.assertIn("Never diagnose", system)
        self.assertIn("emergency", system)
        self.assertEqual(messages[1]["content"], "I have a fever.")

    def test_question_is_stripped(self):
        messages = hc.build_messages("  what is a fever?  ")
        self.assertEqual(messages[1]["content"], "what is a fever?")

    def test_empty_question_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)


class BuildPayloadTest(unittest.TestCase):
    def test_payload_uses_spark_model_and_token_budget(self):
        payload = hc.build_payload("hello")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "system")

    def test_unknown_fields_are_not_added(self):
        payload = hc.build_payload("hello")
        self.assertEqual(
            set(payload),
            {"model", "messages", "max_tokens", "temperature", "stream"},
        )


class ExtractAnswerTest(unittest.TestCase):
    def test_reads_message_content(self):
        raw = json.dumps(_ok_payload("  Rest and drink water.  "))
        self.assertEqual(hc.extract_answer(raw), "Rest and drink water.")

    def test_rejects_non_json(self):
        with self.assertRaises(hc.SparkResponseError):
            hc.extract_answer("<html>oops</html>")

    def test_rejects_unexpected_shape(self):
        with self.assertRaises(hc.SparkResponseError):
            hc.extract_answer(json.dumps({"choices": []}))

    def test_rejects_empty_answer(self):
        with self.assertRaises(hc.SparkResponseError):
            hc.extract_answer(json.dumps(_ok_payload("   ")))


class ChatTest(unittest.TestCase):
    def test_posts_to_chat_completions_and_returns_text(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(_ok_payload("See a clinician if it worsens."))

        with mock.patch.object(hc.urllib.request, "urlopen", fake_urlopen):
            answer = hc.chat("cough for three days", base_url="http://127.0.0.1:30000/v1/")

        self.assertEqual(answer, "See a clinician if it worsens.")
        self.assertEqual(captured["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(captured["body"]["model"], hc.MODEL_ID)
        self.assertEqual(captured["body"]["messages"][1]["content"], "cough for three days")
        self.assertEqual(captured["timeout"], hc.TIMEOUT_SECONDS)

    def test_base_url_can_come_from_environment(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            return _FakeResponse(_ok_payload("ok"))

        with mock.patch.dict(os.environ, {"SPARK_BASE_URL": "http://127.0.0.1:9999/v1"}):
            with mock.patch.object(hc.urllib.request, "urlopen", fake_urlopen):
                hc.chat("hi")

        self.assertEqual(captured["url"], "http://127.0.0.1:9999/v1/chat/completions")

    def test_connection_error_is_wrapped(self):
        def failing_urlopen(request, timeout=None):
            raise hc.urllib.error.URLError("connection refused")

        with mock.patch.object(hc.urllib.request, "urlopen", failing_urlopen):
            with self.assertRaises(hc.SparkEndpointError):
                hc.chat("hi", base_url="http://127.0.0.1:1/v1")

    def test_bad_response_is_wrapped(self):
        def fake_urlopen(request, timeout=None):
            return _FakeResponse({"unexpected": True})

        with mock.patch.object(hc.urllib.request, "urlopen", fake_urlopen):
            with self.assertRaises(hc.SparkResponseError):
                hc.chat("hi")

    def test_empty_question_fails_before_network_call(self):
        with mock.patch.object(hc.urllib.request, "urlopen") as urlopen:
            with self.assertRaises(ValueError):
                hc.chat("  ")
            urlopen.assert_not_called()


class MainTest(unittest.TestCase):
    def test_main_reports_endpoint_errors(self):
        def failing_urlopen(request, timeout=None):
            raise hc.urllib.error.URLError("connection refused")

        with mock.patch.object(hc.urllib.request, "urlopen", failing_urlopen):
            with mock.patch("sys.stdout"), mock.patch("sys.stderr") as stderr:
                code = hc.main(["demo question"])

        self.assertEqual(code, 1)
        self.assertTrue(stderr.write.called)


if __name__ == "__main__":
    unittest.main()
