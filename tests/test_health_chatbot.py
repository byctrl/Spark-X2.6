"""Unit tests for src/health_chatbot.py (no model download required)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_messages_shape(self):
        messages = hc.build_messages("  Demam tiga hari  ")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "Demam tiga hari")

    def test_blank_question_rejected(self):
        for value in ["", "   ", None, 123]:
            with self.assertRaises(ValueError):
                hc.build_messages(value)  # type: ignore[arg-type]

    def test_blank_system_prompt_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("Halo", system_prompt="  ")


class SafetyPromptTest(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_points_to_clinician(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("must not diagnose", prompt)
        self.assertIn("prescribe", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("plain language", prompt)

    def test_model_config(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)
        self.assertTrue(hc.DEMO_QUESTION.strip())


class DisclaimerTest(unittest.TestCase):
    def test_disclaimer_appended(self):
        self.assertEqual(
            hc.format_with_disclaimer("Minum air putih cukup."),
            "Minum air putih cukup.\n\n" + hc.DISCLAIMER,
        )

    def test_disclaimer_not_duplicated(self):
        once = hc.format_with_disclaimer("Minum air putih cukup.")
        self.assertEqual(hc.format_with_disclaimer(once), once)

    def test_empty_reply_returns_disclaimer(self):
        self.assertEqual(hc.format_with_disclaimer(""), hc.DISCLAIMER)
        self.assertEqual(hc.format_with_disclaimer(None), hc.DISCLAIMER)


class AnswerQuestionTest(unittest.TestCase):
    def test_injected_generator_is_used(self):
        seen = []

        def fake_generate(messages):
            seen.extend(messages)
            return "Batuk dan demam bisa punya banyak penyebab."

        answer = hc.answer_question("Batuk dan demam?", generate=fake_generate)

        self.assertEqual(seen[1]["role"], "user")
        self.assertEqual(seen[1]["content"], "Batuk dan demam?")
        self.assertIn("banyak penyebab", answer)
        self.assertIn(hc.DISCLAIMER, answer)

    def test_empty_question_fails_before_model_call(self):
        called = []

        def fake_generate(messages):
            called.append(messages)
            return "tidak terpakai"

        with self.assertRaises(ValueError):
            hc.answer_question("   ", generate=fake_generate)
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
