"""Tests for the Spark-X2.6-4B health-information chatbot demo."""

import os
import sys
import unittest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

import health_chatbot as hc  # noqa: E402


class ConfigTest(unittest.TestCase):
    def test_model_id(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_max_new_tokens(self):
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_keeps_safety_boundaries(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("prescription", prompt)
        self.assertIn("emergency", prompt)
        self.assertIn("clinician", prompt)

    def test_demo_question_is_a_scenario(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


class BuildMessagesTest(unittest.TestCase):
    def test_system_first_user_last(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})
        self.assertEqual(len(messages), 2)

    def test_question_is_stripped(self):
        messages = hc.build_messages("  hello  ")
        self.assertEqual(messages[-1]["content"], "hello")

    def test_empty_question_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_history_is_preserved(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = hc.build_messages("Four days.", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1], history[0])
        self.assertEqual(messages[2], history[1])

    def test_invalid_history_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hi", [{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("hi", ["not a dict"])


class BuildPromptTest(unittest.TestCase):
    def test_prompt_ends_with_assistant_turn(self):
        prompt = hc.build_prompt("When should I see a doctor?")
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertIn(hc.SYSTEM_PROMPT, prompt)
        self.assertIn("When should I see a doctor?", prompt)


class GenerateTest(unittest.TestCase):
    def test_generate_validates_input_before_loading_model(self):
        # Input validation happens before any optional heavy import.
        with self.assertRaises(ValueError):
            hc.generate("")


class MainTest(unittest.TestCase):
    def test_dry_run_returns_zero(self):
        self.assertEqual(hc.main(["--dry-run", "--question", "Is a fever serious?"]), 0)


if __name__ == "__main__":
    unittest.main()
