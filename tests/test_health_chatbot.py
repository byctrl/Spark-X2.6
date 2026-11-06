"""Tests for the minimal Spark-X2.6-4B health chatbot.

Run from the repository root:

    python -m unittest discover -s tests -v

The tests inject a fake generator, so no model download is required.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

import health_chatbot as hc  # noqa: E402


class ModelConfigTest(unittest.TestCase):
    def test_model_id_points_at_spark(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_token_budget_is_bounded(self):
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_sets_safety_boundaries(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never give a diagnosis", prompt)
        self.assertIn("plain language", prompt)
        self.assertIn("emergency", prompt)


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("What does a mild fever usually mean?")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(
            messages[-1],
            {"role": "user", "content": "What does a mild fever usually mean?"},
        )

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi, how can I help?"},
        ]
        messages = hc.build_messages("  next question  ", history=history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[-1]["content"], "next question")

    def test_invalid_history_turns_are_skipped(self):
        history = [
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "kept"},
        ]
        messages = hc.build_messages("q", history=history)
        self.assertEqual([m["role"] for m in messages], ["system", "assistant", "user"])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_non_string_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages(None)


class AnswerTest(unittest.TestCase):
    def test_injected_generator_receives_messages(self):
        seen = {}

        def fake_generate(messages):
            seen["messages"] = messages
            return "  A short plain answer.  "

        out = hc.answer("What is a fever?", generate_fn=fake_generate)
        self.assertEqual(out, "A short plain answer.")
        self.assertEqual(seen["messages"][-1]["content"], "What is a fever?")

    def test_empty_model_output_falls_back(self):
        out = hc.answer("What is a fever?", generate_fn=lambda messages: "   ")
        self.assertEqual(out, hc.FALLBACK_ANSWER)

    def test_none_model_output_falls_back(self):
        out = hc.answer("What is a fever?", generate_fn=lambda messages: None)
        self.assertEqual(out, hc.FALLBACK_ANSWER)

    def test_emergency_notice_is_prepended_for_urgent_wording(self):
        out = hc.answer("I have chest pain right now", generate_fn=lambda messages: "Please rest.")
        self.assertTrue(out.startswith(hc.EMERGENCY_LINE))
        self.assertIn("Please rest.", out)

    def test_emergency_notice_is_not_duplicated(self):
        def fake_generate(messages):
            return hc.EMERGENCY_LINE + "\n\nCall a clinician today."

        out = hc.answer("I cannot breathe", generate_fn=fake_generate)
        self.assertEqual(out.count(hc.EMERGENCY_LINE), 1)

    def test_ordinary_question_has_no_emergency_notice(self):
        out = hc.answer("How much water should I drink a day?", generate_fn=lambda messages: "About six to eight cups.")
        self.assertNotIn(hc.EMERGENCY_LINE, out)

    def test_needs_emergency_notice_detection(self):
        self.assertTrue(hc.needs_emergency_notice("She has a SEVERE BLEEDING wound"))
        self.assertFalse(hc.needs_emergency_notice("I have a mild cough"))
        self.assertFalse(hc.needs_emergency_notice(""))

    def test_demo_question_is_usable(self):
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertEqual(messages[-1]["content"], hc.DEMO_QUESTION)


if __name__ == "__main__":
    unittest.main()
