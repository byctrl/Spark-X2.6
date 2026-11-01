"""Unit tests for the Spark-X2.6-4B health-information chatbot."""

import os
import sys
import unittest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_first_message_is_system_prompt(self):
        messages = hc.build_messages("I have a cough.")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)

    def test_user_question_is_kept(self):
        messages = hc.build_messages("I have a cough.")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "I have a cough.")

    def test_question_is_stripped(self):
        messages = hc.build_messages("  I have a cough.  ")
        self.assertEqual(messages[1]["content"], "I have a cough.")

    def test_empty_question_raises(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")
        with self.assertRaises(ValueError):
            hc.build_messages(None)


class FormatPromptTest(unittest.TestCase):
    def test_prompt_ends_with_assistant_header(self):
        prompt = hc.format_prompt(hc.build_messages("I have a cough."))
        self.assertTrue(prompt.endswith("### Assistant:"))

    def test_prompt_contains_question_and_system_prompt(self):
        prompt = hc.format_prompt(hc.build_messages("I have a cough."))
        self.assertIn("I have a cough.", prompt)
        self.assertIn(hc.SYSTEM_PROMPT, prompt)
        self.assertIn("### System:", prompt)
        self.assertIn("### User:", prompt)


class SafetyTest(unittest.TestCase):
    def test_urgent_symptoms_are_detected(self):
        self.assertTrue(hc.is_urgent("I have chest pain right now"))
        self.assertTrue(hc.is_urgent("He has slurred speech"))
        self.assertTrue(hc.is_urgent("trouble breathing since this morning"))

    def test_non_urgent_text_is_not_flagged(self):
        self.assertFalse(hc.is_urgent("I have a mild cough"))
        self.assertFalse(hc.is_urgent(""))
        self.assertFalse(hc.is_urgent(None))

    def test_notice_mentions_emergency_when_urgent(self):
        self.assertIn("emergency", hc.safety_notice("chest pain").lower())

    def test_notice_mentions_clinician_when_ordinary(self):
        self.assertIn("clinician", hc.safety_notice("I have a cough").lower())

    def test_system_prompt_has_no_diagnosis_rule(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("emergency services", prompt)


class ConfigTest(unittest.TestCase):
    def test_model_id_and_token_budget(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_demo_question_is_usable(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())
        self.assertEqual(len(hc.build_messages(hc.DEMO_QUESTION)), 2)


if __name__ == "__main__":
    unittest.main()
