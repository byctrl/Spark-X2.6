"""Offline tests for the Spark-X2.6-4B health chatbot demo (no network)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot  # noqa: E402


class MessageBuildingTest(unittest.TestCase):
    def test_model_and_token_budget(self):
        self.assertEqual(health_chatbot.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(health_chatbot.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_carries_safety_rules(self):
        prompt = health_chatbot.SYSTEM_PROMPT.lower()
        self.assertIn("diagnosis", prompt)
        self.assertIn("emergency", prompt)
        self.assertIn("professional", prompt)

    def test_build_messages_starts_with_system_prompt(self):
        messages = health_chatbot.build_messages("How do I prepare for a clinic visit?")
        self.assertEqual(messages[0], {"role": "system", "content": health_chatbot.SYSTEM_PROMPT})
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(len(messages), 2)

    def test_build_messages_keeps_valid_history_only(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = health_chatbot.build_messages("question", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            health_chatbot.build_messages("  ")


class PayloadAndParsingTest(unittest.TestCase):
    def test_payload_uses_model_and_budget(self):
        payload = health_chatbot.build_payload("hi")
        self.assertEqual(payload["model"], health_chatbot.MODEL_ID)
        self.assertEqual(payload["max_tokens"], health_chatbot.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1]["content"], "hi")

    def test_parse_response_reads_content(self):
        data = {"choices": [{"message": {"role": "assistant", "content": "  See a clinician.  "}}]}
        self.assertEqual(health_chatbot.parse_response(data), "See a clinician.")

    def test_parse_response_rejects_empty_answer(self):
        with self.assertRaises(RuntimeError):
            health_chatbot.parse_response({"choices": [{"message": {"content": "   "}}]})
        with self.assertRaises(RuntimeError):
            health_chatbot.parse_response({})


if __name__ == "__main__":
    unittest.main()
