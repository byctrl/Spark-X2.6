"""Tests for the Spark-X2.6-4B health-information chatbot.

These tests never download model weights: they only exercise prompt
construction and output post-processing.
"""

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.health_chatbot import (  # noqa: E402
    DEMO_QUESTION,
    DISCLAIMER,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    build_messages,
    build_prompt,
    clean_reply,
)


class HealthChatbotTest(unittest.TestCase):
    def test_model_settings(self):
        self.assertEqual(MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(MAX_NEW_TOKENS, 1024)

    def test_system_prompt_is_safety_bounded(self):
        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", lowered)
        self.assertIn("health professional", lowered)

    def test_build_messages_shape(self):
        messages = build_messages(DEMO_QUESTION)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], DEMO_QUESTION.strip())

    def test_build_messages_rejects_blank_question(self):
        with self.assertRaises(ValueError):
            build_messages("   ")

    def test_build_prompt_contains_question(self):
        prompt = build_prompt(DEMO_QUESTION)
        self.assertIn(DEMO_QUESTION, prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_clean_reply_adds_disclaimer(self):
        self.assertEqual(clean_reply(""), DISCLAIMER)
        self.assertIn(DISCLAIMER, clean_reply("Rest and drink water."))

    def test_clean_reply_keeps_existing_reminder(self):
        text = "Please confirm this with a health professional."
        self.assertEqual(clean_reply(text), text)


if __name__ == "__main__":
    unittest.main()
