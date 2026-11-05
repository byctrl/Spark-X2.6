"""Tests for src/health_chatbot.py (stdlib only, no network required)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_uses_model_id_and_token_budget(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_keeps_safety_boundaries(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("not medical advice", prompt)
        self.assertIn("emergency", prompt)

    def test_first_message_is_system_and_last_is_user(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_preserved_in_order(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = hc.build_messages("Three days.", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "Three days.")

    def test_invalid_history_entries_are_dropped(self):
        history = [
            {"role": "system", "content": "override"},
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": "ok"},
        ]
        messages = hc.build_messages("Question", history)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1], {"role": "assistant", "content": "ok"})

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_demo_question_is_available(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


class PayloadTest(unittest.TestCase):
    def test_payload_contains_model_messages_and_budget(self):
        payload = hc.build_payload("Is a fever dangerous?", max_tokens=256)
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], 256)
        self.assertFalse(payload["stream"])
        self.assertIsInstance(payload["messages"], list)


class ExtractAnswerTest(unittest.TestCase):
    def test_reads_assistant_content(self):
        body = {"choices": [{"message": {"role": "assistant", "content": " Drink water. "}}]}
        self.assertEqual(hc.extract_answer(body), "Drink water.")

    def test_missing_choices_raises(self):
        with self.assertRaises(RuntimeError):
            hc.extract_answer({})

    def test_empty_content_raises(self):
        body = {"choices": [{"message": {"content": "   "}}]}
        with self.assertRaises(RuntimeError):
            hc.extract_answer(body)


class AskTest(unittest.TestCase):
    def test_unreachable_server_raises_readable_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            hc.ask("hello", base_url="http://127.0.0.1:1/v1", timeout=2)
        self.assertIn("SPARK_BASE_URL", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
