"""Tests for the Spark-X2.6-4B health-information chatbot demo."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import health_chatbot  # noqa: E402
from src.health_chatbot import (  # noqa: E402
    DEMO_QUESTION,
    EMERGENCY_NOTICE,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    build_messages,
    build_prompt,
    generate,
    needs_emergency_notice,
)


class ModelConfigTest(unittest.TestCase):
    def test_points_at_spark_model(self):
        self.assertEqual(MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(MAX_NEW_TOKENS, 1024)

    def test_system_prompt_bounds_the_model(self):
        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", lowered)
        self.assertIn("prescribe", lowered)
        self.assertIn("emergency", lowered)
        self.assertIn("plain language", lowered)

    def test_demo_question_is_usable(self):
        self.assertTrue(DEMO_QUESTION.strip())
        self.assertFalse(needs_emergency_notice(DEMO_QUESTION))


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = build_messages("What is a normal resting heart rate?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(
            messages[-1]["content"], "What is a normal resting heart rate?"
        )

    def test_history_is_kept_between_system_and_question(self):
        history = [
            {"role": "user", "content": "I have a headache."},
            {"role": "assistant", "content": "How long has it lasted?"},
            {"role": "tool", "content": "ignored"},
            {"role": "user", "content": ""},
        ]
        messages = build_messages("Two days.", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[-1]["content"], "Two days.")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("   ")

    def test_emergency_question_gets_extra_system_reminder(self):
        messages = build_messages("I have chest pain and cannot breathe")
        self.assertEqual(messages[-2]["role"], "system")
        self.assertEqual(messages[-2]["content"], EMERGENCY_NOTICE)
        self.assertEqual(messages[-1]["role"], "user")

    def test_build_prompt_ends_with_assistant_turn(self):
        prompt = build_prompt(build_messages("Is paracetamol safe with food?"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>"))


class EmergencyGateTest(unittest.TestCase):
    def test_detects_red_flags_case_insensitively(self):
        self.assertTrue(needs_emergency_notice("I have CHEST PAIN"))
        self.assertTrue(needs_emergency_notice("my father had a stroke"))
        self.assertFalse(needs_emergency_notice("I have a mild cough"))
        self.assertFalse(needs_emergency_notice(""))

    def test_emergency_question_never_reaches_the_model(self):
        health_chatbot._PIPELINE = None
        self.addCleanup(setattr, health_chatbot, "_PIPELINE", None)
        self.assertEqual(
            generate("I passed out this morning"),
            EMERGENCY_NOTICE,
        )
        self.assertIsNone(health_chatbot._PIPELINE)


class _FakePipeline:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [{"generated_text": self.reply}]


class GenerateTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, health_chatbot, "_PIPELINE", None)

    def test_uses_spark_messages_and_token_budget(self):
        fake = _FakePipeline("Rest, drink water, and see a clinician if it worsens.")
        health_chatbot._PIPELINE = fake

        answer = generate("I have a mild cough")

        self.assertEqual(
            answer, "Rest, drink water, and see a clinician if it worsens."
        )
        messages, kwargs = fake.calls[0]
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[-1]["content"], "I have a mild cough")
        self.assertEqual(kwargs["max_new_tokens"], MAX_NEW_TOKENS)
        self.assertFalse(kwargs["do_sample"])

    def test_handles_chat_style_output(self):
        fake = _FakePipeline([{"role": "assistant", "content": "See a clinician."}])
        health_chatbot._PIPELINE = fake
        self.assertEqual(generate("I have a mild fever"), "See a clinician.")


if __name__ == "__main__":
    unittest.main()
