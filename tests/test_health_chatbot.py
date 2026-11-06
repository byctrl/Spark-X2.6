"""Unit tests for the health chatbot wiring (no model download required)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.health_chatbot import (  # noqa: E402
    DEMO_QUESTION,
    DISCLAIMER,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    ChatError,
    build_messages,
    chat,
)


class RecordingGenerator:
    def __init__(self, reply="General information about fever and cough."):
        self.reply = reply
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.reply


class ConstantsTest(unittest.TestCase):
    def test_model_id(self):
        self.assertEqual(MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_max_new_tokens(self):
        self.assertEqual(MAX_NEW_TOKENS, 1024)

    def test_system_prompt_is_safety_bounded(self):
        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("diagnose", lowered)
        self.assertIn("prescribe", lowered)
        self.assertIn("clinician", lowered)

    def test_demo_question_is_usable(self):
        self.assertTrue(DEMO_QUESTION.strip())

    def test_disclaimer_says_not_medical_advice(self):
        self.assertIn("not medical advice", DISCLAIMER.lower())


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = build_messages("How long does a fever usually last?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")

    def test_question_is_trimmed(self):
        messages = build_messages("  what is a normal temperature?  ")
        self.assertEqual(messages[1]["content"], "what is a normal temperature?")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ChatError):
            build_messages("   ")

    def test_non_string_question_is_rejected(self):
        with self.assertRaises(ChatError):
            build_messages(None)


class ChatTest(unittest.TestCase):
    def test_answer_includes_disclaimer(self):
        gen = RecordingGenerator("Rest and fluids are often suggested.")
        self.assertIn(DISCLAIMER, chat("what helps with a cough?", gen))

    def test_generator_receives_messages(self):
        gen = RecordingGenerator()
        chat("is a fever of 38 C serious?", gen)
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual(gen.calls[0][0]["role"], "system")
        self.assertEqual(gen.calls[0][1]["content"], "is a fever of 38 C serious?")

    def test_empty_model_reply_is_rejected(self):
        with self.assertRaises(ChatError):
            chat("any question", RecordingGenerator("   "))

    def test_dict_reply_is_accepted(self):
        gen = RecordingGenerator({"content": "Ask your clinician about next steps."})
        answer = chat("should I worry?", gen)
        self.assertIn("Ask your clinician", answer)


if __name__ == "__main__":
    unittest.main()
