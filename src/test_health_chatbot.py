"""Unit tests for src/health_chatbot.py (no network, no model download)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_is_first_and_question_is_last(self):
        messages = hc.build_messages("  I have a cough  ")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "I have a cough")

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_system_prompt_forbids_diagnosis_and_prescribing(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("do not prescribe", prompt)

    def test_payload_uses_spark_model_and_token_bound(self):
        payload = hc.build_payload("what is a fever?")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(payload["messages"], hc.build_messages("what is a fever?"))


class AskTest(unittest.TestCase):
    def test_ask_sends_payload_and_returns_trimmed_content(self):
        captured = {}

        def fake_transport(url, payload, api_key):
            captured["url"] = url
            captured["payload"] = payload
            captured["api_key"] = api_key
            return {"choices": [{"message": {"content": "  See a clinician.  "}}]}

        answer = hc.ask(
            "I have a cough",
            base_url="http://127.0.0.1:30000/v1/",
            api_key="EMPTY",
            transport=fake_transport,
        )

        self.assertEqual(answer, "See a clinician.")
        self.assertEqual(captured["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(captured["api_key"], "EMPTY")

    def test_malformed_response_raises_runtime_error(self):
        for bad in ({}, {"choices": []}, {"choices": [{"message": {}}]}):
            with self.assertRaises(RuntimeError):
                hc.extract_answer(bad)

    def test_empty_answer_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            hc.extract_answer({"choices": [{"message": {"content": "   "}}]})


class SafetyTest(unittest.TestCase):
    def test_emergency_terms_detected(self):
        self.assertTrue(hc.needs_emergency_notice("I have chest pain"))
        self.assertTrue(hc.needs_emergency_notice("Trouble Breathing at night"))
        self.assertFalse(hc.needs_emergency_notice("I have a mild dry cough"))

    def test_demo_question_is_not_flagged_as_emergency(self):
        self.assertFalse(hc.needs_emergency_notice(hc.DEMO_QUESTION))


if __name__ == "__main__":
    unittest.main()
