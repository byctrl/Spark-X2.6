"""Offline tests for src/health_chatbot.py (no model, no network)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)

    def test_question_is_last_user_turn(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_kept_in_order_and_filtered(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello, I am not a clinician."},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("What is a fever?", history=history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_safety_limits_are_stated_in_prompt(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("not a clinician", prompt)


class PayloadAndReplyTest(unittest.TestCase):
    def test_payload_targets_spark_model(self):
        payload = hc.build_payload(hc.build_messages("What is a fever?"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)
        self.assertFalse(payload["stream"])

    def test_parse_reply_reads_content(self):
        data = {"choices": [{"message": {"role": "assistant", "content": " Drink water. "}}]}
        self.assertEqual(hc.parse_reply(data), "Drink water.")

    def test_parse_reply_rejects_empty_or_bad_shape(self):
        with self.assertRaises(ValueError):
            hc.parse_reply({"choices": [{"message": {"content": ""}}]})
        with self.assertRaises(ValueError):
            hc.parse_reply({"unexpected": True})


if __name__ == "__main__":
    unittest.main()
