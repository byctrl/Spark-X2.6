"""Offline tests for src/health_chatbot.py (no model or network needed)."""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)

    def test_question_is_last_user_turn(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_kept_and_filtered(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("next", history=history)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class BuildPayloadTest(unittest.TestCase):
    def test_payload_uses_model_and_token_budget(self):
        payload = hc.build_payload("question")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])

    def test_payload_can_override_model(self):
        payload = hc.build_payload("question", model="spark-served")
        self.assertEqual(payload["model"], "spark-served")


class ParseResponseTest(unittest.TestCase):
    def test_extracts_content(self):
        data = {"choices": [{"message": {"role": "assistant", "content": " hi "}}]}
        self.assertEqual(hc.parse_response(data), "hi")

    def test_rejects_missing_choices(self):
        with self.assertRaises(ValueError):
            hc.parse_response({})

    def test_rejects_empty_content(self):
        data = {"choices": [{"message": {"content": "  "}}]}
        with self.assertRaises(ValueError):
            hc.parse_response(data)


class EnsureSafeTest(unittest.TestCase):
    def test_disclaimer_added_when_no_referral(self):
        result = hc.ensure_safe("Rest and drink water.")
        self.assertTrue(result.startswith("Rest and drink water."))
        self.assertIn(hc.DISCLAIMER, result)

    def test_referral_is_not_duplicated(self):
        text = "Please see a doctor if this gets worse."
        self.assertEqual(hc.ensure_safe(text), text)

    def test_empty_answer_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.ensure_safe("")


if __name__ == "__main__":
    unittest.main()
