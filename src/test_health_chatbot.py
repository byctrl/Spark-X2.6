#!/usr/bin/env python3
"""Offline unit tests for src/health_chatbot.py (no network required)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTests(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)

    def test_user_question_is_last_turn(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_question_is_stripped(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[-1]["content"], "What is a fever?")

    def test_history_is_kept_between_prompt_and_question(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
            {"role": "tool", "content": "ignored"},
            {"role": "assistant", "content": "   "},
        ]
        messages = hc.build_messages("Three days.", history=history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"], "I have a cough.")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "Three days.")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class SafetyPromptTests(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_prescription(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("not medical advice", prompt)
        self.assertIn("emergency", prompt)

    def test_demo_question_exists(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


class BuildPayloadTests(unittest.TestCase):
    def test_payload_uses_spark_model_and_token_budget(self):
        messages = hc.build_messages("What is a fever?")
        payload = hc.build_payload(messages)
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"], messages)


class ExtractAnswerTests(unittest.TestCase):
    def test_reads_and_strips_content(self):
        response = {"choices": [{"message": {"content": "  Drink water.  "}}]}
        self.assertEqual(hc.extract_answer(response), "Drink water.")

    def test_empty_content_raises(self):
        with self.assertRaises(RuntimeError):
            hc.extract_answer({"choices": [{"message": {"content": "   "}}]})

    def test_malformed_response_raises(self):
        with self.assertRaises(RuntimeError):
            hc.extract_answer({"error": "boom"})


class ChatTests(unittest.TestCase):
    def test_chat_posts_to_chat_completions_and_returns_answer(self):
        captured = {}

        def fake_transport(url, payload, timeout):
            captured["url"] = url
            captured["payload"] = payload
            captured["timeout"] = timeout
            return {"choices": [{"message": {"content": "See a clinician if it lasts."}}]}

        answer = hc.chat(
            "What is a fever?", base_url="http://127.0.0.1:30000/v1/", transport=fake_transport
        )

        self.assertEqual(answer, "See a clinician if it lasts.")
        self.assertEqual(captured["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(captured["payload"]["messages"][0]["role"], "system")
        self.assertGreater(captured["timeout"], 0)

    def test_chat_uses_env_base_url(self):
        captured = {}

        def fake_transport(url, payload, timeout):
            captured["url"] = url
            return {"choices": [{"message": {"content": "ok"}}]}

        previous = os.environ.get("SPARK_BASE_URL")
        os.environ["SPARK_BASE_URL"] = "http://localhost:8000/v1"
        try:
            hc.chat("What is a fever?", transport=fake_transport)
        finally:
            if previous is None:
                os.environ.pop("SPARK_BASE_URL", None)
            else:
                os.environ["SPARK_BASE_URL"] = previous

        self.assertEqual(captured["url"], "http://localhost:8000/v1/chat/completions")

    def test_chat_rejects_empty_question(self):
        def fake_transport(url, payload, timeout):
            raise AssertionError("transport must not be called")

        with self.assertRaises(ValueError):
            hc.chat("  ", transport=fake_transport)


if __name__ == "__main__":
    unittest.main()
