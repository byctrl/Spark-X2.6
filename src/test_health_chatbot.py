"""Tests for the Spark-X2.6-4B health-information chatbot demo.

These tests never download or run the model: they cover prompt construction
and the safety boundary only. Run from the repository root with:

    python -m unittest discover -s src -p "test_*.py" -v
"""

import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(len(messages), 2)
        self.assertEqual(
            messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT}
        )
        self.assertEqual(
            messages[-1], {"role": "user", "content": "What is a fever?"}
        )

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        messages = hc.build_messages("next question", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_demo_question_is_non_empty(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


class SafetyTest(unittest.TestCase):
    def test_system_prompt_forbids_diagnosis_and_prescribing(self):
        text = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", text)
        self.assertIn("prescribe", text)
        self.assertIn("qualified clinician", text)

    def test_red_flag_is_detected_case_insensitively(self):
        self.assertTrue(hc.has_red_flag("I have Chest Pain right now"))
        self.assertFalse(hc.has_red_flag("What is a mild fever?"))

    def test_red_flag_short_circuits_the_model(self):
        calls = []

        def fake_generator(messages):
            calls.append(messages)
            return "the model should not be asked"

        reply = hc.answer("I cannot breathe properly", generator=fake_generator)
        self.assertEqual(reply, hc.EMERGENCY_REPLY)
        self.assertEqual(calls, [])

    def test_emergency_reply_points_to_emergency_care(self):
        self.assertIn("emergency", hc.EMERGENCY_REPLY.lower())

    def test_normal_question_uses_trimmed_model_output(self):
        seen = {}

        def fake_generator(messages):
            seen["messages"] = messages
            return "  Fever is a raised body temperature. Confirm with a clinician.  "

        reply = hc.answer("What is a fever?", generator=fake_generator)
        self.assertEqual(
            reply,
            "Fever is a raised body temperature. Confirm with a clinician.",
        )
        self.assertEqual(seen["messages"][-1]["content"], "What is a fever?")

    def test_blank_model_output_falls_back(self):
        reply = hc.answer("What is a fever?", generator=lambda messages: "   ")
        self.assertEqual(reply, hc.FALLBACK_REPLY)


if __name__ == "__main__":
    unittest.main()
