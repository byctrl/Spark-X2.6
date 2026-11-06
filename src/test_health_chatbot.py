"""Unit tests for the Spark-X2.6-4B health chatbot demo (no network needed)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_and_user_turns(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "What is a fever?")

    def test_question_is_stripped(self):
        messages = hc.build_messages("  cough  ")
        self.assertEqual(messages[1]["content"], "cough")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class SafetyPromptTest(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_points_to_clinician(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("emergency", prompt)

    def test_demo_question_is_present(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


class PayloadTest(unittest.TestCase):
    def test_payload_fields(self):
        payload = hc.build_payload("hello")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertIsInstance(payload["temperature"], float)

    def test_max_tokens_override(self):
        payload = hc.build_payload("hello", max_tokens=64)
        self.assertEqual(payload["max_tokens"], 64)


class ExtractAnswerTest(unittest.TestCase):
    def test_reads_content(self):
        body = {"choices": [{"message": {"content": "  Drink water.  "}}]}
        self.assertEqual(hc.extract_answer(body), "Drink water.")

    def test_missing_choices_is_an_error(self):
        with self.assertRaises(ValueError):
            hc.extract_answer({})

    def test_empty_content_is_an_error(self):
        body = {"choices": [{"message": {"content": "   "}}]}
        with self.assertRaises(ValueError):
            hc.extract_answer(body)


if __name__ == "__main__":
    unittest.main()
