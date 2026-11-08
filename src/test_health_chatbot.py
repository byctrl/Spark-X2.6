"""Offline unit tests for the Spark-X2.6-4B health-education chatbot.

These tests never touch the network: they cover prompt assembly, request
payload shape, safety wording and response validation only.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc  # noqa: E402


class MessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("How much water should I drink?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "How much water should I drink?")

    def test_question_is_stripped(self):
        messages = hc.build_messages("  fever  \n")
        self.assertEqual(messages[-1]["content"], "fever")

    def test_blank_question_rejected(self):
        for value in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(value)


class SafetyTest(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_prescription(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", lowered)
        self.assertIn("prescribe", lowered)
        self.assertIn("clinician", lowered)
        self.assertIn("emergency", lowered)


class PayloadTest(unittest.TestCase):
    def test_payload_targets_repo_model_and_token_budget(self):
        payload = hc.build_payload(hc.DEMO_QUESTION)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["messages"][1]["content"], hc.DEMO_QUESTION)
        json.dumps(payload)  # must be JSON serialisable

    def test_invalid_token_budget_rejected(self):
        for value in (0, -5, None, "1024"):
            with self.assertRaises(ValueError):
                hc.build_payload("hi", max_tokens=value)


class ExtractAnswerTest(unittest.TestCase):
    def test_reads_and_strips_content(self):
        response = {"choices": [{"message": {"role": "assistant", "content": " Drink water. "}}]}
        self.assertEqual(hc.extract_answer(response), "Drink water.")

    def test_rejects_malformed_responses(self):
        bad_responses = [
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": "   "}}]},
            {"choices": [{"message": {"content": None}}]},
            {"choices": "nope"},
            None,
        ]
        for response in bad_responses:
            with self.assertRaises(ValueError):
                hc.extract_answer(response)


class ConfigTest(unittest.TestCase):
    def test_demo_question_is_a_real_scenario(self):
        self.assertGreater(len(hc.DEMO_QUESTION), 40)
        self.assertTrue(hc.DEFAULT_BASE_URL.startswith("http"))


if __name__ == "__main__":
    unittest.main()
