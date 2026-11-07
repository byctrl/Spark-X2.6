"""Tests for the minimal Spark-X2.6-4B health chatbot demo (issue #16)."""

import os
import sys
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import health_chatbot as hc  # noqa: E402


class ConstantsTest(unittest.TestCase):
    def test_model_and_token_budget(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_is_safety_bounded(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("emergency", prompt)
        self.assertIn("clinician", prompt)


class BuildMessagesTest(unittest.TestCase):
    def test_roles_and_content(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1], {"role": "user", "content": "What is a fever?"})

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_emergency_note_is_added(self):
        messages = hc.build_messages("My father has chest pain right now")
        self.assertIn(hc.EMERGENCY_NOTE, messages[0]["content"])

    def test_routine_question_has_no_emergency_note(self):
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertNotIn(hc.EMERGENCY_NOTE, messages[0]["content"])

    def test_looks_like_emergency(self):
        self.assertTrue(hc.looks_like_emergency("I cannot breathe"))
        self.assertFalse(hc.looks_like_emergency(hc.DEMO_QUESTION))
        self.assertFalse(hc.looks_like_emergency(""))


class PayloadTest(unittest.TestCase):
    def test_payload_shape(self):
        payload = hc.build_payload("How much water should I drink?")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][1]["role"], "user")


class ExtractReplyTest(unittest.TestCase):
    def test_extracts_content(self):
        response = {"choices": [{"message": {"content": "  Drink water.  "}}]}
        self.assertEqual(hc.extract_reply(response), "Drink water.")

    def test_missing_choices(self):
        with self.assertRaises(hc.ChatbotError):
            hc.extract_reply({"choices": []})

    def test_empty_answer(self):
        with self.assertRaises(hc.ChatbotError):
            hc.extract_reply({"choices": [{"message": {"content": ""}}]})

    def test_non_dict_response(self):
        with self.assertRaises(hc.ChatbotError):
            hc.extract_reply("not-json")


class CallSparkTest(unittest.TestCase):
    def test_unreachable_endpoint_raises_chatbot_error(self):
        with mock.patch(
            "health_chatbot.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with self.assertRaises(hc.ChatbotError) as ctx:
                hc.call_spark("What is a fever?")
        self.assertIn("Cannot reach the Spark endpoint", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
