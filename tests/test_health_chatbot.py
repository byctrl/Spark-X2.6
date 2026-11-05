"""Tests for the Spark-X2.6-4B health-information chatbot demo.

These tests exercise prompt construction and safety wording only; they must
not require torch / transformers or a downloaded model.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.health_chatbot import (  # noqa: E402
    DEMO_QUESTION,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SAFETY_NOTE,
    SYSTEM_PROMPT,
    build_messages,
    build_prompt,
)


class FakeTokenizer:
    """Minimal stand-in exposing a chat template."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        return rendered + ("\nassistant:" if add_generation_prompt else "")


class NoTemplateTokenizer:
    """Stand-in without a chat template, to check the fallback path."""


class TestHealthChatbot(unittest.TestCase):
    def test_model_id_is_spark_x2_6_4b(self):
        self.assertEqual(MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_max_new_tokens_is_positive_int(self):
        self.assertIsInstance(MAX_NEW_TOKENS, int)
        self.assertGreater(MAX_NEW_TOKENS, 0)

    def test_system_prompt_contains_safety_rules(self):
        lowered = SYSTEM_PROMPT.lower()
        for phrase in ("diagnose", "prescribe", "professional"):
            self.assertIn(phrase, lowered)

    def test_safety_note_states_limits(self):
        lowered = SAFETY_NOTE.lower()
        self.assertIn("not a diagnosis", lowered)
        self.assertIn("professional", lowered)

    def test_demo_question_is_non_empty(self):
        self.assertTrue(DEMO_QUESTION.strip())

    def test_build_messages_structure(self):
        messages = build_messages(DEMO_QUESTION)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], DEMO_QUESTION.strip())

    def test_build_messages_trims_whitespace(self):
        messages = build_messages("  demam ringan  ")
        self.assertEqual(messages[1]["content"], "demam ringan")

    def test_build_messages_accepts_custom_system_prompt(self):
        messages = build_messages("halo", system_prompt="custom")
        self.assertEqual(messages[0]["content"], "custom")

    def test_build_messages_rejects_empty_input(self):
        for bad in ("", "   ", None, 123):
            with self.assertRaises(ValueError):
                build_messages(bad)

    def test_build_messages_rejects_empty_system_prompt(self):
        with self.assertRaises(ValueError):
            build_messages("halo", system_prompt="   ")

    def test_build_prompt_uses_chat_template(self):
        prompt = build_prompt(build_messages("apa itu demam?"), FakeTokenizer())
        self.assertIn("system: ", prompt)
        self.assertIn("user: apa itu demam?", prompt)
        self.assertTrue(prompt.rstrip().endswith("assistant:"))

    def test_build_prompt_falls_back_without_template(self):
        prompt = build_prompt(build_messages("apa itu demam?"), NoTemplateTokenizer())
        self.assertIn("system: ", prompt)
        self.assertIn("user: apa itu demam?", prompt)
        self.assertTrue(prompt.rstrip().endswith("assistant:"))

    def test_build_prompt_tolerates_template_errors(self):
        class BrokenTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise ValueError("no template")

        prompt = build_prompt(build_messages("halo"), BrokenTokenizer())
        self.assertIn("user: halo", prompt)

    def test_import_does_not_pull_in_transformers(self):
        # Prompt helpers must stay usable without the ML stack installed.
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
