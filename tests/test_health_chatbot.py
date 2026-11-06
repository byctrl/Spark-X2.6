"""Tests for the Spark-X2.6-4B health chatbot demo (no model needed)."""

import os
import sys
import unittest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import health_chatbot as hc  # noqa: E402


class HealthChatbotTest(unittest.TestCase):
    def test_model_id(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_max_new_tokens_is_bounded(self):
        self.assertIsInstance(hc.MAX_NEW_TOKENS, int)
        self.assertGreater(hc.MAX_NEW_TOKENS, 0)
        self.assertLessEqual(hc.MAX_NEW_TOKENS, 4096)

    def test_system_prompt_is_safety_bounded(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("professional", prompt)
        self.assertIn("emergency", prompt)

    def test_build_messages_shape(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_filtered(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": "ignored"},
            {"role": "assistant", "content": "   "},
            {"role": "assistant", "content": "hi there"},
        ]
        messages = hc.build_messages("next question", history=history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1], {"role": "user", "content": "hello"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "hi there"})

    def test_empty_question_rejected(self):
        for question in ("", "   ", None, 123):
            with self.assertRaises(ValueError):
                hc.build_messages(question)

    def test_payload(self):
        payload = hc.build_payload("is this normal?")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1]["content"], "is this normal?")

    def test_extract_reply(self):
        good = {"choices": [{"message": {"content": "  some guidance  "}}]}
        self.assertEqual(hc.extract_reply(good), "some guidance")
        with self.assertRaises(ValueError):
            hc.extract_reply({"choices": []})
        with self.assertRaises(ValueError):
            hc.extract_reply({"choices": [{"message": {"content": "   "}}]})
        with self.assertRaises(ValueError):
            hc.extract_reply({})


if __name__ == "__main__":
    unittest.main()
