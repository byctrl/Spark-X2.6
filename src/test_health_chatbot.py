"""Tests for the minimal Spark-X2.6-4B health chatbot.

Run from the repository root:
    python -m unittest discover -s src -p "test_*.py"
"""

import unittest
from unittest import mock

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("hello")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1], {"role": "user", "content": "hello"})

    def test_question_is_stripped(self):
        self.assertEqual(hc.build_messages("  hi  ")[1]["content"], "hi")

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_non_string_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages(None)

    def test_long_question_is_truncated(self):
        messages = hc.build_messages("a" * (hc.MAX_QUESTION_CHARS + 50))
        self.assertEqual(len(messages[1]["content"]), hc.MAX_QUESTION_CHARS)


class PayloadTest(unittest.TestCase):
    def test_payload_targets_spark_and_uses_token_budget(self):
        payload = hc.build_payload("hi")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(len(payload["messages"]), 2)


class SafetyPromptTest(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_requires_referral(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("clinician", prompt)


class ExtractAnswerTest(unittest.TestCase):
    def test_extracts_and_strips_content(self):
        response = {"choices": [{"message": {"content": "  Drink water.  "}}]}
        self.assertEqual(hc.extract_answer(response), "Drink water.")

    def test_rejects_malformed_responses(self):
        bad_responses = [
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": "   "}}]},
            {"choices": [{"message": {"content": None}}]},
        ]
        for response in bad_responses:
            with self.assertRaises(hc.ModelError):
                hc.extract_answer(response)


class ChatTest(unittest.TestCase):
    def test_transport_receives_built_payload(self):
        captured = {}

        def fake_transport(payload):
            captured.update(payload)
            return {"choices": [{"message": {"content": "See a doctor."}}]}

        self.assertEqual(hc.chat("cough", transport=fake_transport), "See a doctor.")
        self.assertEqual(captured["model"], hc.MODEL_ID)
        self.assertEqual(captured["messages"][0]["content"], hc.SYSTEM_PROMPT)

    def test_env_base_url_is_used_for_http_call(self):
        seen = {}

        def fake_post(url, payload, timeout=hc.DEFAULT_TIMEOUT):
            seen["url"] = url
            return {"choices": [{"message": {"content": "ok"}}]}

        with mock.patch.dict("os.environ", {"SPARK_BASE_URL": "http://localhost:9999/v1/"}):
            with mock.patch.object(hc, "_http_post_json", fake_post):
                self.assertEqual(hc.chat("hi"), "ok")
        self.assertEqual(seen["url"], "http://localhost:9999/v1/chat/completions")

    def test_default_base_url_is_used_without_env(self):
        seen = {}

        def fake_post(url, payload, timeout=hc.DEFAULT_TIMEOUT):
            seen["url"] = url
            return {"choices": [{"message": {"content": "ok"}}]}

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.object(hc, "_http_post_json", fake_post):
                hc.chat("hi")
        self.assertEqual(seen["url"], hc.DEFAULT_BASE_URL + "/chat/completions")

    def test_transport_errors_propagate(self):
        def broken_transport(payload):
            raise hc.ModelError("boom")

        with self.assertRaises(hc.ModelError):
            hc.chat("hi", transport=broken_transport)


class MainTest(unittest.TestCase):
    def test_reports_model_error_without_crashing(self):
        with mock.patch.object(hc, "chat", side_effect=hc.ModelError("offline")):
            self.assertEqual(hc.main(["hi"]), 1)

    def test_demo_question_is_used_without_arguments(self):
        answers = {}

        def fake_chat(question, **kwargs):
            answers["question"] = question
            return "answer"

        with mock.patch.object(hc, "chat", fake_chat):
            self.assertEqual(hc.main([]), 0)
        self.assertEqual(answers["question"], hc.DEMO_QUESTION)


if __name__ == "__main__":
    unittest.main()
