"""Tests for the Spark-X2.6-4B health-information chatbot (no model needed)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_and_trimmed_question(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(
            messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT}
        )
        self.assertEqual(
            messages[1], {"role": "user", "content": "What is a fever?"}
        )

    def test_system_prompt_is_safety_bounded(self):
        self.assertIn("NOT diagnose", hc.SYSTEM_PROMPT)
        self.assertIn("emergency services", hc.SYSTEM_PROMPT)

    def test_demo_question_is_present(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class ChatTest(unittest.TestCase):
    def test_payload_and_answer(self):
        captured = {}

        def fake_transport(url, payload, timeout):
            captured["url"] = url
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "  Plain answer.  "}}]}

        answer = hc.chat(
            "What is a fever?",
            base_url="http://localhost:9/v1",
            transport=fake_transport,
        )

        self.assertEqual(answer, "Plain answer.")
        self.assertEqual(captured["url"], "http://localhost:9/v1/chat/completions")
        payload = captured["payload"]
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(payload["messages"][1]["content"], "What is a fever?")

    def test_malformed_response_raises(self):
        def fake_transport(url, payload, timeout):
            return {"unexpected": True}

        with self.assertRaises(RuntimeError):
            hc.chat("hello", transport=fake_transport)

    def test_empty_answer_raises(self):
        def fake_transport(url, payload, timeout):
            return {"choices": [{"message": {"content": "   "}}]}

        with self.assertRaises(RuntimeError):
            hc.chat("hello", transport=fake_transport)


if __name__ == "__main__":
    unittest.main()
