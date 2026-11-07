"""Offline unit tests for the Spark-X2.6-4B health chatbot demo."""

import io
import json
import unittest
import urllib.error
from unittest import mock

import health_chatbot as hc


class FakeResponse(io.BytesIO):
    """Minimal stand-in for http.client.HTTPResponse."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def body_with(answer):
    return json.dumps({"choices": [{"message": {"content": answer}}]}).encode("utf-8")


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  I have a fever  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "I have a fever"})

    def test_history_is_kept_in_order_and_junk_is_dropped(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
            "not-a-dict",
        ]
        messages = hc.build_messages("next", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])


class ExtractAnswerTest(unittest.TestCase):
    def test_reads_content(self):
        self.assertEqual(hc.extract_answer(json.loads(body_with(" rest "))), "rest")

    def test_rejects_bad_shapes(self):
        for bad in ({}, {"choices": []}, {"choices": [{"message": {}}]}, None):
            with self.assertRaises(ValueError):
                hc.extract_answer(bad)

    def test_rejects_empty_answer(self):
        with self.assertRaises(ValueError):
            hc.extract_answer(json.loads(body_with("   ")))


class DisclaimerTest(unittest.TestCase):
    def test_adds_once(self):
        once = hc.with_disclaimer("Drink water.")
        self.assertTrue(once.endswith(hc.DISCLAIMER))
        self.assertEqual(hc.with_disclaimer(once), once)


class AskSparkTest(unittest.TestCase):
    def test_sends_model_and_messages(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(body_with("Rest and see a clinician."))

        with mock.patch("health_chatbot.urllib.request.urlopen", fake_urlopen):
            answer = hc.ask_spark("cough", base_url="http://127.0.0.1:30000/v1/")

        self.assertEqual(answer, "Rest and see a clinician.")
        self.assertEqual(captured["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(captured["payload"]["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(captured["payload"]["messages"][-1]["content"], "cough")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.ask_spark("   ")

    def test_main_reports_unreachable_model(self):
        def boom(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        with mock.patch("health_chatbot.urllib.request.urlopen", boom):
            with mock.patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(hc.main(["fever"]), 1)


if __name__ == "__main__":
    unittest.main()
