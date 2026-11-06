"""Offline tests for the Spark-X2.6-4B health chatbot (no model, no network)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.health_chatbot import (  # noqa: E402
    DEMO_QUESTION,
    MODEL_ID,
    SYSTEM_PROMPT,
    build_messages,
    build_payload,
    parse_response,
)


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = build_messages(DEMO_QUESTION)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], DEMO_QUESTION.strip())

    def test_history_sits_between_system_prompt_and_question(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        messages = build_messages("  what is a fever?  ", history=history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"], "hi")
        self.assertEqual(messages[2]["content"], "hello")
        self.assertEqual(messages[-1]["content"], "what is a fever?")

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                build_messages(bad)

    def test_invalid_history_turn_is_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("q", history=[{"role": "tool", "content": "x"}])
        with self.assertRaises(ValueError):
            build_messages("q", history=[{"role": "user", "content": "  "}])


class BuildPayloadTest(unittest.TestCase):
    def test_payload_targets_spark_model(self):
        payload = build_payload(build_messages("q"))
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertFalse(payload["stream"])
        self.assertGreater(payload["max_tokens"], 0)

    def test_empty_messages_are_rejected(self):
        with self.assertRaises(ValueError):
            build_payload([])


class ParseResponseTest(unittest.TestCase):
    def test_returns_trimmed_content(self):
        data = {"choices": [{"message": {"role": "assistant", "content": "  see a clinician  "}}]}
        self.assertEqual(parse_response(data), "see a clinician")

    def test_rejects_empty_or_malformed_responses(self):
        bad_payloads = (
            {},
            {"choices": []},
            {"choices": [{"message": {"content": "   "}}]},
            None,
        )
        for bad in bad_payloads:
            with self.assertRaises(ValueError):
                parse_response(bad)


if __name__ == "__main__":
    unittest.main()
