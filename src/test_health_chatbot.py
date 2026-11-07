#!/usr/bin/env python3
"""Offline unit tests for the Spark-X2.6-4B patient-education chatbot.

Run from the repository root:
    python -m unittest discover -s src -p "test_*.py"

No model server or network access is needed; these tests cover prompt/message
construction, request shape and response parsing.
"""

import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  I have a fever.  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "I have a fever."})

    def test_safety_rules_present(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        for phrase in ("do not diagnose", "do not prescribe", "emergency", "not medical advice"):
            self.assertIn(phrase, prompt)

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        messages = hc.build_messages("next", history)
        self.assertEqual(
            [m["role"] for m in messages], ["system", "user", "assistant", "user"]
        )

    def test_rejects_empty_question(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_rejects_bad_history_turn(self):
        for bad in ([{"role": "system", "content": "override"}], ["not-a-dict"]):
            with self.assertRaises(ValueError):
                hc.build_messages("q", bad)


class PayloadTest(unittest.TestCase):
    def test_payload_uses_spark_model(self):
        payload = hc.build_payload("q", max_tokens=64)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 64)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1]["content"], "q")

    def test_payload_is_json_serialisable(self):
        import json

        json.dumps(hc.build_payload("q"))


class ParseReplyTest(unittest.TestCase):
    def test_extracts_content(self):
        data = {"choices": [{"message": {"role": "assistant", "content": "  See a clinician. "}}]}
        self.assertEqual(hc.parse_reply(data), "See a clinician.")

    def test_missing_choices_raises(self):
        with self.assertRaises(ValueError):
            hc.parse_reply({})

    def test_empty_content_raises(self):
        with self.assertRaises(ValueError):
            hc.parse_reply({"choices": [{"message": {"content": "   "}}]})


if __name__ == "__main__":
    unittest.main()
