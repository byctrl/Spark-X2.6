"""Tests for the minimal Spark-X2.6-4B health-knowledge chatbot."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_is_first_message(self):
        messages = hc.build_messages("what is a fever?")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})

    def test_question_is_last_message(self):
        messages = hc.build_messages("  what is a fever?  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "what is a fever?"})

    def test_history_is_kept_before_the_new_question(self):
        history = [
            {"role": "user", "content": "I have a cough"},
            {"role": "assistant", "content": "Since when?"},
        ]
        messages = hc.build_messages("three days", history=history)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "three days")

    def test_blank_history_turns_are_dropped(self):
        messages = hc.build_messages("hello", history=[{"role": "user", "content": "   "}])
        self.assertEqual(len(messages), 2)

    def test_invalid_history_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hello", history=[{"role": "system", "content": "x"}])

    def test_empty_question_is_rejected(self):
        for question in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(question)

    def test_system_prompt_forbids_diagnosis_and_dosage(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("diagnos", prompt)
        self.assertIn("dosage", prompt)
        self.assertIn("emergency", prompt)


class PayloadAndParsingTest(unittest.TestCase):
    def test_payload_uses_model_and_token_budget(self):
        payload = hc.build_payload("hi")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])

    def test_parse_response_extracts_content(self):
        body = {"choices": [{"message": {"role": "assistant", "content": "  Drink water.  "}}]}
        self.assertEqual(hc.parse_response(body), "Drink water.")

    def test_parse_response_rejects_empty_content(self):
        body = {"choices": [{"message": {"content": "   "}}]}
        with self.assertRaises(ValueError):
            hc.parse_response(body)

    def test_parse_response_rejects_bad_shape(self):
        for body in ({}, {"choices": []}, {"choices": [{}]}, None):
            with self.assertRaises(ValueError):
                hc.parse_response(body)


if __name__ == "__main__":
    unittest.main()
