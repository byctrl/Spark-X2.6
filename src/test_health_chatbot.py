"""Offline tests for the Spark-X2.6-4B health-information chatbot demo."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc  # noqa: E402


class MessageTests(unittest.TestCase):
    def test_system_prompt_keeps_safety_boundary(self):
        self.assertIn("Never diagnose", hc.SYSTEM_PROMPT)
        self.assertIn("clinician", hc.SYSTEM_PROMPT)

    def test_messages_shape(self):
        messages = hc.build_messages("  how much water should I drink?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "how much water should I drink?"})
        self.assertEqual(len(messages), 2)

    def test_history_is_preserved(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        messages = hc.build_messages("q", history=history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_blank_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_bad_history_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("q", history=[{"role": "system", "content": "x"}])


class PayloadTests(unittest.TestCase):
    def test_payload_uses_spark_model_id(self):
        payload = hc.build_payload(hc.DEMO_QUESTION)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])
        json.dumps(payload)  # must stay JSON serialisable

    def test_payload_token_budget_override(self):
        payload = hc.build_payload("q", max_tokens=256)
        self.assertEqual(payload["max_tokens"], 256)


class ParseTests(unittest.TestCase):
    def test_parse_content(self):
        data = {"choices": [{"message": {"content": "  rest and fluids  "}}]}
        self.assertEqual(hc.parse_response(data), "rest and fluids")

    def test_parse_empty_content_fails(self):
        with self.assertRaises(ValueError):
            hc.parse_response({"choices": [{"message": {"content": "   "}}]})

    def test_parse_missing_choices_fails(self):
        with self.assertRaises(ValueError):
            hc.parse_response({})


class SafetyTests(unittest.TestCase):
    def test_footer_added_once(self):
        text = hc.add_safety_footer("Rest and drink fluids.")
        self.assertIn("not a diagnosis", text)
        self.assertEqual(text, hc.add_safety_footer(text))

    def test_urgent_wording_escalates(self):
        text = hc.add_safety_footer("Call a doctor.", question="I have chest pain")
        self.assertIn("emergency services", text)


if __name__ == "__main__":
    unittest.main()
