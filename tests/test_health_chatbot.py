"""Unit tests for src/health_chatbot.py (no model download required)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("Is a mild fever normal?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)

    def test_question_is_last_user_turn_and_stripped(self):
        messages = hc.build_messages("  I have a cough  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "I have a cough"})

    def test_history_is_copied_in_order(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
        ]
        messages = hc.build_messages("I have a cough", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1], history[0])
        self.assertEqual(messages[2], history[1])
        self.assertEqual(len(history), 2)

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_malformed_history_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("Hi", [{"role": "system", "content": "x"}])


class AnswerTest(unittest.TestCase):
    def test_fake_generator_receives_messages_and_budget(self):
        seen = {}

        def fake_generate(messages, max_new_tokens=None):
            seen["messages"] = messages
            seen["max_new_tokens"] = max_new_tokens
            return "  Drink fluids and rest; see a clinician.  "

        result = hc.answer("Fever after travel", generate=fake_generate)

        self.assertEqual(seen["max_new_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(seen["messages"][0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(result["reply"], "Drink fluids and rest; see a clinician.")
        self.assertEqual(result["question"], "Fever after travel")
        self.assertEqual(result["disclaimer"], hc.DISCLAIMER)

    def test_custom_budget_is_forwarded(self):
        seen = {}

        def fake_generate(messages, max_new_tokens=None):
            seen["max_new_tokens"] = max_new_tokens
            return "ok"

        hc.answer("question", generate=fake_generate, max_new_tokens=64)
        self.assertEqual(seen["max_new_tokens"], 64)

    def test_empty_model_output_is_an_error(self):
        with self.assertRaises(RuntimeError):
            hc.answer("question", generate=lambda messages, max_new_tokens=None: "  ")


class ConstantsTest(unittest.TestCase):
    def test_model_and_budget(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_safety_text_forbids_diagnosis_and_prescription(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", lowered)
        self.assertIn("medication", lowered)
        self.assertIn("clinician", lowered)

    def test_demo_question_and_disclaimer_exist(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())
        self.assertIn("not medical advice", hc.DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
