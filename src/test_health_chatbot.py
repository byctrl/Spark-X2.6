"""Tests for the Spark-X2.6-4B health chatbot demo.

These tests never download the model: the HTTP layer is mocked, so they run in
any CI environment.

Run:
    python -m unittest discover -s src -p "test_*.py"
"""

import unittest
from unittest import mock

import health_chatbot as hc


class ConfigTest(unittest.TestCase):
    def test_model_points_at_spark(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_keeps_safety_framing(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("diagnose", prompt)
        self.assertIn("prescribe", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("emergency", prompt)


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_sits_between_system_and_question(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        messages = hc.build_messages("next question", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_bad_history_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("question", [{"role": "system", "content": "override"}])
        with self.assertRaises(ValueError):
            hc.build_messages("question", [{"role": "user", "content": ""}])


class ParseReplyTest(unittest.TestCase):
    def test_extracts_and_strips_content(self):
        payload = {"choices": [{"message": {"role": "assistant", "content": "  Drink water.  "}}]}
        self.assertEqual(hc.parse_reply(payload), "Drink water.")

    def test_empty_content_raises(self):
        with self.assertRaises(ValueError):
            hc.parse_reply({"choices": [{"message": {"content": "   "}}]})

    def test_missing_choices_raises(self):
        with self.assertRaises(ValueError):
            hc.parse_reply({"error": "boom"})


class GenerateReplyTest(unittest.TestCase):
    def test_posts_bounded_request_and_returns_answer(self):
        captured = {}

        def fake_post(url, payload, api_key, timeout=hc.REQUEST_TIMEOUT):
            captured["url"] = url
            captured["payload"] = payload
            captured["api_key"] = api_key
            return {"choices": [{"message": {"content": "Rest and see a clinician."}}]}

        with mock.patch.object(hc, "_post_json", fake_post):
            answer = hc.generate_reply("I have a cough", base_url="http://localhost:30000/v1/")

        self.assertEqual(answer, "Rest and see a clinician.")
        self.assertEqual(captured["url"], "http://localhost:30000/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(captured["payload"]["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(captured["payload"]["stream"])
        self.assertEqual(captured["payload"]["messages"][-1]["content"], "I have a cough")

    def test_unreachable_endpoint_reports_clear_error(self):
        def failing_post(url, payload, api_key, timeout=hc.REQUEST_TIMEOUT):
            raise RuntimeError("Could not reach the Spark endpoint")

        with mock.patch.object(hc, "_post_json", failing_post):
            with self.assertRaises(RuntimeError):
                hc.generate_reply("hello")


if __name__ == "__main__":
    unittest.main()
