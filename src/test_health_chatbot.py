"""Offline tests for src/health_chatbot.py (no network, no model download)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class PromptTests(unittest.TestCase):
    def test_model_and_token_budget(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_messages_have_system_and_user_roles(self):
        messages = hc.build_messages("What is heat stroke?")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "What is heat stroke?")

    def test_system_prompt_forbids_diagnosis(self):
        prompt = hc.build_messages("anything")[0]["content"].lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("not medical advice", prompt)

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", "\n\t"):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_question_whitespace_is_collapsed(self):
        messages = hc.build_messages("  cough\n\n and   fever ")
        self.assertEqual(messages[1]["content"], "cough and fever")

    def test_long_question_is_truncated(self):
        messages = hc.build_messages("a" * (hc.MAX_INPUT_CHARS + 500))
        self.assertEqual(len(messages[1]["content"]), hc.MAX_INPUT_CHARS)

    def test_payload_is_openai_compatible(self):
        payload = hc.build_payload("How do I prevent dehydration?")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertIsInstance(payload["messages"], list)

    def test_demo_question_exists(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())
        self.assertEqual(len(hc.build_messages(hc.DEMO_QUESTION)), 2)


class ResponseParsingTests(unittest.TestCase):
    def test_extract_answer(self):
        response = {"choices": [{"message": {"role": "assistant", "content": " Stay hydrated. "}}]}
        self.assertEqual(hc.extract_answer(response), "Stay hydrated.")

    def test_extract_answer_rejects_empty_choices(self):
        with self.assertRaises(ValueError):
            hc.extract_answer({"choices": []})

    def test_extract_answer_rejects_blank_content(self):
        with self.assertRaises(ValueError):
            hc.extract_answer({"choices": [{"message": {"content": "   "}}]})

    def test_extract_answer_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            hc.extract_answer(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
