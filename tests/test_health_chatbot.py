"""Offline tests for src/health_chatbot.py (no model server required)."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTests(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_system_prompt_forbids_diagnosis_and_prescribing(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("do not prescribe", prompt)

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_is_appended_before_question(self):
        history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        messages = hc.build_messages("next", history=history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_invalid_history_role_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("q", history=[{"role": "system", "content": "x"}])


class PayloadTests(unittest.TestCase):
    def test_payload_uses_model_and_token_budget(self):
        payload = hc.build_payload(hc.build_messages("q"))
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])


class ResponseTests(unittest.TestCase):
    def test_parse_chat_completion(self):
        payload = {"choices": [{"message": {"role": "assistant", "content": "  Drink water.  "}}]}
        self.assertEqual(hc.parse_chat_completion(payload), "Drink water.")

    def test_parse_rejects_empty_answer(self):
        with self.assertRaises(ValueError):
            hc.parse_chat_completion({"choices": [{"message": {"content": "   "}}]})

    def test_parse_rejects_missing_choices(self):
        with self.assertRaises(ValueError):
            hc.parse_chat_completion({})

    def test_disclaimer_is_appended_once(self):
        once = hc.ensure_disclaimer("General information.")
        twice = hc.ensure_disclaimer(once)
        self.assertEqual(once, twice)
        self.assertIn(hc.DISCLAIMER, once)
        self.assertTrue(once.startswith("General information."))


if __name__ == "__main__":
    unittest.main()
