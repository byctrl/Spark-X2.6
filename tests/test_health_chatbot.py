"""Offline tests for the Spark-X2.6-4B health chatbot (no model weights needed)."""

import os
import sys
import unittest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

import health_chatbot  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_uses_spark_model_id(self):
        self.assertEqual(health_chatbot.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(health_chatbot.MAX_NEW_TOKENS, 1024)

    def test_returns_system_then_user(self):
        messages = health_chatbot.build_messages("What is a fever?")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "What is a fever?")
        self.assertIn("Spark-X2.6-4B", messages[0]["content"])

    def test_system_prompt_blocks_diagnosis_and_points_to_clinician(self):
        system = health_chatbot.build_messages("Is this serious?")[0]["content"]
        self.assertIn("not diagnose", system)
        self.assertIn("clinician", system)
        self.assertIn("emergency", system)

    def test_chinese_language_hint(self):
        system = health_chatbot.build_messages("发烧怎么办", language="zh")[0]["content"]
        self.assertIn("Simplified Chinese", system)

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            health_chatbot.build_messages("   ")


def _FakeTokenizer(**kwargs):

    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
        return "TEMPLATE|" + messages[-1]["content"]

    return type("FakeTokenizer", (), {"apply_chat_template": staticmethod(apply_chat_template)})()


class RenderPromptTest(unittest.TestCase):
    def test_fallback_prompt_contains_question_and_assistant_turn(self):
        messages = health_chatbot.build_messages("How much water is enough?")
        prompt = health_chatbot.render_prompt(messages)
        self.assertIn("How much water is enough?", prompt)
        self.assertTrue(prompt.rstrip().endswith("assistant:"))

    def test_tokenizer_template_is_used_when_available(self):
        messages = health_chatbot.build_messages("Hello")
        prompt = health_chatbot.render_prompt(messages, tokenizer=_FakeTokenizer())
        self.assertEqual(prompt, "TEMPLATE|Hello")


class DemoTest(unittest.TestCase):
    def test_demo_question_is_present(self):
        self.assertIn("doctor", health_chatbot.DEMO_QUESTION)
        self.assertEqual(len(health_chatbot.build_messages(health_chatbot.DEMO_QUESTION)), 2)

    def test_generate_rejects_empty_question_before_loading_model(self):
        with self.assertRaises(ValueError):
            health_chatbot.generate("")


if __name__ == "__main__":
    unittest.main()
