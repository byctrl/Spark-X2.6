"""Offline tests for src/health_chatbot.py (no model download required)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_then_question(self):
        messages = hc.build_messages("Apa itu demam?")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1], {"role": "user", "content": "Apa itu demam?"})

    def test_history_is_kept_and_filtered(self):
        history = [
            {"role": "user", "content": "Halo"},
            {"role": "assistant", "content": "Halo, ada yang bisa saya bantu?"},
            {"role": "system", "content": "ignored"},
            {"role": "assistant", "content": "   "},
        ]
        messages = hc.build_messages("Batuk tiga hari", history=history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "Batuk tiga hari")

    def test_empty_question_rejected(self):
        for question in (None, "", "   "):
            with self.assertRaises(ValueError):
                hc.build_messages(question)


class BuildPayloadTest(unittest.TestCase):
    def test_payload_uses_spark_model(self):
        payload = hc.build_payload("Apa itu demam?")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1]["role"], "user")

    def test_payload_token_override(self):
        payload = hc.build_payload("Apa itu demam?", max_new_tokens=64)
        self.assertEqual(payload["max_tokens"], 64)


class ParseResponseTest(unittest.TestCase):
    def test_answer_gets_disclaimer(self):
        data = {"choices": [{"message": {"content": "  Demam ringan bisa dipantau di rumah.  "}}]}
        answer = hc.parse_response(data)
        self.assertTrue(answer.startswith("Demam ringan"))
        self.assertIn(hc.DISCLAIMER, answer)

    def test_disclaimer_not_duplicated(self):
        text = "Minum cairan cukup. This is general health information, not medical advice."
        self.assertEqual(hc.with_disclaimer(text), text)

    def test_malformed_response_raises(self):
        for data in ({}, {"choices": []}, {"choices": [{}]}, None):
            with self.assertRaises(hc.ChatbotError):
                hc.parse_response(data)

    def test_empty_answer_raises(self):
        data = {"choices": [{"message": {"content": "   "}}]}
        with self.assertRaises(hc.ChatbotError):
            hc.parse_response(data)


class AskTest(unittest.TestCase):
    def test_unreachable_endpoint_raises_chatbot_error(self):
        with self.assertRaises(hc.ChatbotError):
            hc.ask("Apa itu demam?", base_url="http://127.0.0.1:1/v1", timeout=2.0)

    def test_demo_question_is_usable(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())
        self.assertEqual(hc.build_messages(hc.DEMO_QUESTION)[-1]["content"], hc.DEMO_QUESTION)


if __name__ == "__main__":
    unittest.main()
