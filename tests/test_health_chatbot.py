"""Dependency-free checks for src/health_chatbot.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("Apa itu demam?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Apa itu demam?"})
        self.assertEqual(len(messages), 2)

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "Saya batuk."},
            {"role": "assistant", "content": "Sejak kapan?"},
        ]
        messages = hc.build_messages("  Tiga hari.  ", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[-1]["content"], "Tiga hari.")

    def test_invalid_history_entries_are_skipped(self):
        messages = hc.build_messages(
            "Halo",
            [{"role": "tool", "content": "x"}, {"role": "user", "content": "  "}],
        )
        self.assertEqual(len(messages), 2)

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")
        with self.assertRaises(ValueError):
            hc.build_messages(None)

    def test_model_settings(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)


class SafetyTest(unittest.TestCase):
    def test_system_prompt_forbids_diagnosis_and_prescription(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        for phrase in ("must not diagnose", "prescribe", "emergency", "not a substitute"):
            self.assertIn(phrase, prompt)

    def test_disclaimer_and_referral_are_added(self):
        reply = hc.ensure_safety_reply("Rest and drink water.")
        self.assertTrue(reply.startswith("Rest and drink water."))
        self.assertIn(hc.DISCLAIMER, reply)
        self.assertIn(hc.REFERRAL, reply)

    def test_safety_appendix_is_idempotent(self):
        once = hc.ensure_safety_reply("Rest.")
        twice = hc.ensure_safety_reply(once)
        self.assertEqual(once, twice)
        self.assertEqual(twice.count(hc.DISCLAIMER), 1)


if __name__ == "__main__":
    unittest.main()
