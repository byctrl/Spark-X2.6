"""Offline tests for the Spark-X2.6-4B health-education chatbot."""

import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("Should I see a clinic?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Should I see a clinic?"})

    def test_safety_rules_present(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("clinician", prompt)

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_is_preserved(self):
        messages = hc.build_messages(
            "and then?",
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hello, how can I help?"},
                {"role": "tool", "content": "ignored"},
            ],
        )
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[-2]["role"], "assistant")


class PayloadTest(unittest.TestCase):
    def test_payload_uses_model_and_token_budget(self):
        payload = hc.build_payload(hc.build_messages("q"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])


class ParseResponseTest(unittest.TestCase):
    def test_extracts_answer(self):
        data = {"choices": [{"message": {"content": "  See a clinician.  "}}]}
        self.assertEqual(hc.parse_response(data), "See a clinician.")

    def test_missing_choices_raises(self):
        with self.assertRaises(ValueError):
            hc.parse_response({})

    def test_empty_answer_raises(self):
        with self.assertRaises(ValueError):
            hc.parse_response({"choices": [{"message": {"content": "   "}}]})


class CliTest(unittest.TestCase):
    def test_dry_run_prints_payload(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            code = hc.main(["--demo", "--dry-run"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["messages"][-1]["content"], hc.DEMO_QUESTION)


if __name__ == "__main__":
    unittest.main()
