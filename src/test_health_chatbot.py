"""Offline tests for src/health_chatbot.py (no network, no model needed)."""

import json
import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_and_question_are_sent(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "What is a fever?")

    def test_question_is_trimmed(self):
        self.assertEqual(hc.build_messages("  hi  ")[1]["content"], "hi")

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_system_prompt_keeps_safety_boundaries(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("must not diagnose", prompt)
        self.assertIn("clinician", prompt)


class BuildPayloadTest(unittest.TestCase):
    def test_payload_targets_spark_and_is_bounded(self):
        payload = hc.build_payload(hc.build_messages("hi"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        json.dumps(payload)

    def test_empty_messages_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_payload([])


class ExtractTextTest(unittest.TestCase):
    def test_reads_assistant_content(self):
        response = {"choices": [{"message": {"content": "  Drink water.  "}}]}
        self.assertEqual(hc.extract_text(response), "Drink water.")

    def test_rejects_malformed_or_empty_content(self):
        bad_responses = [
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {"content": "   "}}]},
            {"choices": [{"message": {"content": None}}]},
            "not-a-dict",
        ]
        for response in bad_responses:
            with self.assertRaises(ValueError):
                hc.extract_text(response)


if __name__ == "__main__":
    unittest.main()
