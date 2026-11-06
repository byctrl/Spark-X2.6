"""Tests for the minimal Spark-X2.6-4B health-information chatbot.

These tests use only the standard library and never download the model.
Run from the repository root:

    python -m unittest discover -s src -p "test_*.py"
"""

import unittest

import health_chatbot as hc


class ModelConfigTest(unittest.TestCase):
    def test_model_id(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_max_new_tokens(self):
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_demo_question_present(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())

    def test_system_prompt_has_safety_guardrails(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("not diagnose", prompt)
        self.assertIn("prescribe", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("emergency", prompt)


class BuildMessagesTest(unittest.TestCase):
    def test_structure(self):
        messages = hc.build_messages("  Apa itu demam?  ")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "Apa itu demam?")

    def test_history_is_included_and_filtered(self):
        history = [
            {"role": "user", "content": "Halo"},
            {"role": "assistant", "content": "Halo, ada yang bisa dibantu?"},
            {"role": "tool", "content": "dihapus"},
            {"role": "user", "content": ""},
        ]
        messages = hc.build_messages("Lanjutkan", history)
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "Lanjutkan")

    def test_history_is_bounded(self):
        history = [
            {"role": "user", "content": "pesan %d" % i}
            for i in range(hc.MAX_HISTORY_TURNS + 5)
        ]
        messages = hc.build_messages("terakhir", history)
        self.assertEqual(len(messages), hc.MAX_HISTORY_TURNS + 2)


class ValidateQuestionTest(unittest.TestCase):
    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            hc.validate_question("   ")

    def test_rejects_non_string(self):
        with self.assertRaises(ValueError):
            hc.validate_question(None)

    def test_rejects_too_long(self):
        with self.assertRaises(ValueError):
            hc.validate_question("a" * (hc.MAX_QUESTION_CHARS + 1))

    def test_strips_whitespace(self):
        self.assertEqual(hc.validate_question("  hi  "), "hi")


class DisclaimerTest(unittest.TestCase):
    def test_appends_note(self):
        result = hc.ensure_disclaimer("Minum air yang cukup.")
        self.assertTrue(result.startswith("Minum air yang cukup."))
        self.assertIn(hc.EMERGENCY_NOTE, result)

    def test_is_idempotent(self):
        once = hc.ensure_disclaimer("Batasi aktivitas.")
        twice = hc.ensure_disclaimer(once)
        self.assertEqual(once, twice)

    def test_empty_reply_returns_note(self):
        self.assertEqual(hc.ensure_disclaimer(""), hc.EMERGENCY_NOTE)


class PromptFallbackTest(unittest.TestCase):
    def test_prompt_text_contains_roles(self):
        messages = hc.build_messages("Apa itu demam?")
        prompt = hc.build_prompt_text(messages)
        self.assertIn("system:", prompt)
        self.assertIn("user: Apa itu demam?", prompt)
        self.assertTrue(prompt.endswith("assistant:"))


if __name__ == "__main__":
    unittest.main()
