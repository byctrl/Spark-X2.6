"""Tests for the Spark-X2.6-4B health-information chatbot demo.

These tests never download or load the model.
"""

import os
import sys
import unittest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import health_chatbot as hc  # noqa: E402


class BuildMessagesTests(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)

    def test_question_is_stripped(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[1]["content"], "What is a fever?")

    def test_extra_context_is_appended(self):
        messages = hc.build_messages("What is a fever?", "I am 30 years old.")
        self.assertIn("What is a fever?", messages[1]["content"])
        self.assertIn("I am 30 years old.", messages[1]["content"])

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_blank_context_is_ignored(self):
        messages = hc.build_messages("What is a fever?", "   ")
        self.assertNotIn("Extra context", messages[1]["content"])


class BuildPromptTests(unittest.TestCase):
    def test_prompt_contains_roles_and_question(self):
        prompt = hc.build_prompt(hc.build_messages("What is a fever?"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertTrue(prompt.rstrip().endswith("<|assistant|>"))
        self.assertIn("What is a fever?", prompt)


class ConfigurationTests(unittest.TestCase):
    def test_model_id(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_max_new_tokens(self):
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_safety_framing(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        for phrase in ("never give a diagnosis", "clinician", "not a substitute"):
            self.assertIn(phrase, prompt)

    def test_demo_question_is_usable(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertEqual(messages[1]["content"], hc.DEMO_QUESTION.strip())


class LazyImportTests(unittest.TestCase):
    def test_transformers_not_imported_at_module_load(self):
        self.assertNotIn("transformers", sys.modules)


if __name__ == "__main__":
    unittest.main()
