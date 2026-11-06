"""Tests for src/health_chatbot.py (no model server required)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import health_chatbot  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_includes_system_prompt_and_question(self):
        messages = health_chatbot.build_messages("  what is a fever?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], health_chatbot.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "what is a fever?"})

    def test_appends_history_between_prompt_and_question(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = health_chatbot.build_messages("Three days.", history)
        self.assertEqual([m["role"] for m in messages], [
            "system", "user", "assistant", "user"
        ])
        self.assertEqual(messages[-1]["content"], "Three days.")

    def test_rejects_empty_question(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                health_chatbot.build_messages(bad)

    def test_rejects_malformed_history_turn(self):
        with self.assertRaises(ValueError):
            health_chatbot.build_messages("hi", [{"role": "system", "content": "x"}])

    def test_prompt_forbids_diagnosis_and_points_to_clinicians(self):
        prompt = health_chatbot.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("emergency", prompt)


class ChatTest(unittest.TestCase):
    def test_sends_model_id_and_token_budget(self):
        captured = {}

        def transport(payload):
            captured.update(payload)
            return {"choices": [{"message": {"content": " Ask a clinician. "}}]}

        answer = health_chatbot.chat("fever after travel", transport=transport)

        self.assertEqual(answer, "Ask a clinician.")
        self.assertEqual(captured["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(captured["max_tokens"], 1024)
        self.assertEqual(captured["messages"][-1]["content"], "fever after travel")

    def test_concatenates_content_parts(self):
        def transport(payload):
            return {"choices": [{"message": {"content": [
                {"type": "text", "text": "Part one. "},
                {"type": "text", "text": "Part two."},
            ]}}]}

        self.assertEqual(health_chatbot.chat("hi", transport=transport), "Part one. Part two.")

    def test_empty_answer_raises(self):
        def transport(payload):
            return {"choices": [{"message": {"content": "   "}, "finish_reason": "length"}]}

        with self.assertRaises(RuntimeError) as ctx:
            health_chatbot.chat("hi", transport=transport)
        self.assertIn("length", str(ctx.exception))

    def test_malformed_response_raises(self):
        with self.assertRaises(RuntimeError):
            health_chatbot.chat("hi", transport=lambda payload: {"error": "boom"})

    def test_demo_question_runs_end_to_end(self):
        def transport(payload):
            return {"choices": [{"message": {"content": "General information only."}}]}

        answer = health_chatbot.chat(health_chatbot.DEMO_QUESTION, transport=transport)
        self.assertTrue(answer)


if __name__ == "__main__":
    unittest.main()
