"""Unit tests for src/health_chatbot.py (no model download required)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("How much water should an adult drink?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(len(messages), 2)

    def test_user_question_is_last_and_stripped(self):
        messages = hc.build_messages("  hello  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "hello"})

    def test_history_is_preserved_in_order(self):
        history = [
            {"role": "user", "content": "What is a fever?"},
            {"role": "assistant", "content": "A fever is a raised body temperature."},
        ]
        messages = hc.build_messages("Is 38.3 C high?", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_invalid_history_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hi", [{"role": "system", "content": "x"}])

    def test_history_missing_content_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hi", [{"role": "user"}])


class ConstantsTest(unittest.TestCase):
    def test_model_id(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_token_budget(self):
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_prompt_keeps_safety_boundaries(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        for phrase in ("not diagnose", "clinician", "emergency"):
            self.assertIn(phrase, prompt)

    def test_demo_question_is_usable(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())
        self.assertEqual(len(hc.build_messages(hc.DEMO_QUESTION)), 2)

    def test_safety_reminder_is_plain_text(self):
        self.assertIn("not medical advice", hc.SAFETY_REMINDER.lower())


if __name__ == "__main__":
    unittest.main()
