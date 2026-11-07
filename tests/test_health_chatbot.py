"""Offline unit tests for src/health_chatbot.py (no model, no network)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.health_chatbot import (  # noqa: E402
    DEMO_QUESTION,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    build_messages,
    build_payload,
    extract_answer,
)


class SystemPromptTest(unittest.TestCase):
    def test_forbids_diagnosis_and_points_to_a_clinician(self):
        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("diagnos", lowered)
        self.assertIn("clinician", lowered)
        self.assertIn("emergency", lowered)


class BuildMessagesTest(unittest.TestCase):
    def test_messages_shape(self):
        messages = build_messages("  I have a cough  ")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "I have a cough")

    def test_empty_question_rejected(self):
        for value in ("", "   "):
            with self.assertRaises(ValueError):
                build_messages(value)


class BuildPayloadTest(unittest.TestCase):
    def test_payload_targets_the_model(self):
        payload = build_payload(DEMO_QUESTION)
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertEqual(payload["max_tokens"], MAX_NEW_TOKENS)
        self.assertEqual(len(payload["messages"]), 2)


class ExtractAnswerTest(unittest.TestCase):
    def test_reads_and_strips_content(self):
        data = {"choices": [{"message": {"content": "  See a clinician.  "}}]}
        self.assertEqual(extract_answer(data), "See a clinician.")

    def test_rejects_malformed_or_empty_responses(self):
        for data in ({}, {"choices": []}, {"choices": [{"message": {"content": "  "}}]}):
            with self.assertRaises(RuntimeError):
                extract_answer(data)


if __name__ == "__main__":
    unittest.main()
