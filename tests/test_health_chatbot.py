"""Tests for the minimal Spark-X2.6-4B health-information chatbot."""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import health_chatbot as hc


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class ModelPinningTest(unittest.TestCase):
    def test_model_id_and_token_budget(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_keeps_safety_rules(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never give a diagnosis", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("urgent care", prompt)


class MessageBuildingTest(unittest.TestCase):
    def test_system_prompt_is_first_and_question_is_last(self):
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": hc.DEMO_QUESTION})

    def test_history_turns_are_kept_in_order(self):
        history = [
            {"role": "user", "content": "I have a fever."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = hc.build_messages("Five days.", history=history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1], history[0])
        self.assertEqual(messages[2], history[1])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_invalid_history_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hi", history=[{"role": "system", "content": "x"}])

    def test_request_body_uses_model_and_budget(self):
        body = hc.build_request(hc.DEMO_QUESTION)
        self.assertEqual(body["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(body["max_tokens"], 1024)
        self.assertFalse(body["stream"])


class ResponseParsingTest(unittest.TestCase):
    def test_extract_text_strips_content(self):
        payload = {"choices": [{"message": {"content": "  Drink water.  "}}]}
        self.assertEqual(hc.extract_text(payload), "Drink water.")

    def test_extract_text_rejects_empty_answer(self):
        payload = {"choices": [{"message": {"content": "   "}}]}
        with self.assertRaises(ValueError):
            hc.extract_text(payload)

    def test_extract_text_rejects_unexpected_shape(self):
        with self.assertRaises(ValueError):
            hc.extract_text({"error": "boom"})


class CompletionTest(unittest.TestCase):
    def test_chat_completion_posts_to_local_endpoint(self):
        payload = {"choices": [{"message": {"content": "Rest and watch for fever."}}]}
        with mock.patch.object(
            hc.urllib.request, "urlopen", return_value=_FakeResponse(payload)
        ) as urlopen:
            text = hc.chat_completion("I have a cough.", base_url="http://127.0.0.1:30000/v1/")

        self.assertEqual(text, "Rest and watch for fever.")
        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["model"], hc.MODEL_ID)
        self.assertEqual(sent["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(sent["messages"][0]["role"], "system")


class SafetyNoticeTest(unittest.TestCase):
    def test_answer_appends_safety_notice(self):
        with mock.patch.object(hc, "chat_completion", return_value="Rest and drink water."):
            reply = hc.answer("I have a cough.")
        self.assertTrue(reply.startswith("Rest and drink water."))
        self.assertIn(hc.SAFETY_NOTICE, reply)

    def test_answer_does_not_duplicate_safety_notice(self):
        with mock.patch.object(
            hc, "chat_completion", return_value="Rest.\n\n" + hc.SAFETY_NOTICE
        ):
            reply = hc.answer("I have a cough.")
        self.assertEqual(reply.count(hc.SAFETY_NOTICE), 1)


class CliTest(unittest.TestCase):
    def test_demo_run_prints_answer(self):
        with mock.patch.object(hc, "answer", return_value="Mocked advice."):
            out = io.StringIO()
            with redirect_stdout(out):
                code = hc.main(["--demo"])
        self.assertEqual(code, 0)
        printed = out.getvalue()
        self.assertIn("Mocked advice.", printed)
        self.assertIn(hc.MODEL_ID, printed)

    def test_unreachable_endpoint_exits_non_zero(self):
        with mock.patch.object(
            hc, "answer", side_effect=hc.urllib.error.URLError("refused")
        ):
            out = io.StringIO()
            with redirect_stdout(out):
                code = hc.main(["question?", "--base-url", "http://127.0.0.1:1/v1"])
        self.assertEqual(code, 1)
        self.assertIn("SPARK_BASE_URL", out.getvalue())


if __name__ == "__main__":
    unittest.main()
