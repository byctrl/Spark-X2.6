"""Offline tests for the Spark-X2.6-4B health-information chatbot.

Run from the repository root:
    python -m unittest discover -s src -p "test_*.py"

No model, no network and no extra dependencies are required.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class TestConstants(unittest.TestCase):
    def test_model_id(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_token_budget_is_bounded(self):
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_forbids_diagnosis(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("not diagnose", prompt)
        self.assertIn("not medical", prompt)

    def test_demo_question_is_present(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


class TestBuildMessages(unittest.TestCase):
    def test_system_first_and_user_last(self):
        messages = hc.build_messages("  I have a cough  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "I have a cough"})

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")
        with self.assertRaises(ValueError):
            hc.build_messages(None)

    def test_history_is_preserved_in_order(self):
        history = [
            {"role": "user", "content": "I have a fever"},
            {"role": "assistant", "content": "How high is the fever?"},
        ]
        messages = hc.build_messages("38.5 C", history=history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "I have a fever")

    def test_rejects_malformed_history(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hi", history=[{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("hi", history=[{"role": "user", "content": ""}])


class TestPayloadAndParsing(unittest.TestCase):
    def test_payload_bounds(self):
        payload = hc.build_payload(hc.build_messages("hello"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])

    def test_parse_response_trims(self):
        data = {"choices": [{"message": {"role": "assistant", "content": "  drink water  "}}]}
        self.assertEqual(hc.parse_response(data), "drink water")

    def test_parse_response_rejects_empty(self):
        data = {"choices": [{"message": {"content": "   "}}]}
        with self.assertRaises(ValueError):
            hc.parse_response(data)

    def test_parse_response_rejects_bad_shape(self):
        for data in ({}, {"choices": []}, {"error": "boom"}, None):
            with self.assertRaises(ValueError):
                hc.parse_response(data)


class TestEmergencyGuard(unittest.TestCase):
    def test_flags_emergency_keywords(self):
        self.assertTrue(hc.needs_emergency_notice("my father has chest pain"))
        self.assertTrue(hc.needs_emergency_notice("I cannot breathe properly"))
        self.assertTrue(hc.needs_emergency_notice("heavy bleeding since last night"))

    def test_ignores_ordinary_questions(self):
        self.assertFalse(hc.needs_emergency_notice("I have a mild cough"))
        self.assertFalse(hc.needs_emergency_notice(""))
        self.assertFalse(hc.needs_emergency_notice(None))


if __name__ == "__main__":
    unittest.main()
