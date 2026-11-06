"""Offline tests for the Spark-X2.6-4B health-information chatbot."""

import unittest

from health_chatbot import (
    DEMO_QUESTION,
    DISCLAIMER,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    URGENT_NOTICE,
    build_messages,
    format_reply,
    needs_urgent_care,
)


class ModelConfigTest(unittest.TestCase):
    def test_model_id_points_at_spark(self):
        self.assertEqual(MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_token_budget_is_bounded(self):
        self.assertGreater(MAX_NEW_TOKENS, 0)
        self.assertLessEqual(MAX_NEW_TOKENS, 2048)

    def test_system_prompt_refuses_diagnosis(self):
        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", lowered)
        self.assertIn("clinician", lowered)


class BuildMessagesTest(unittest.TestCase):
    def test_system_then_user(self):
        messages = build_messages(DEMO_QUESTION)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], DEMO_QUESTION)
        self.assertEqual(len(messages), 2)

    def test_history_is_kept_and_filtered(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi, how can I help?"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
            "not-a-dict",
        ]
        messages = build_messages("what is a fever?", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                build_messages(bad)


class SafetyTest(unittest.TestCase):
    def test_detects_emergency_phrasing(self):
        self.assertTrue(needs_urgent_care("I have chest pain since this morning"))
        self.assertTrue(needs_urgent_care("She cannot breathe properly"))

    def test_normal_question_is_not_flagged(self):
        self.assertFalse(needs_urgent_care(DEMO_QUESTION))
        self.assertFalse(needs_urgent_care(""))

    def test_reply_always_carries_disclaimer(self):
        reply = format_reply(DEMO_QUESTION, "Rest and drink water.")
        self.assertIn("Rest and drink water.", reply)
        self.assertTrue(reply.endswith(DISCLAIMER))
        self.assertNotIn(URGENT_NOTICE, reply)

    def test_emergency_reply_leads_with_urgent_notice(self):
        reply = format_reply("I have chest pain", "Let us discuss it.")
        self.assertTrue(reply.startswith(URGENT_NOTICE))
        self.assertTrue(reply.endswith(DISCLAIMER))


if __name__ == "__main__":
    unittest.main()
