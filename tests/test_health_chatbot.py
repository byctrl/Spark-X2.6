"""Offline unit tests for the Spark-X2.6-4B health chatbot demo."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.health_chatbot import (
    DEMO_QUESTION,
    MAX_NEW_TOKENS,
    MAX_QUESTION_CHARS,
    MODEL_ID,
    SYSTEM_PROMPT,
    build_messages,
    build_payload,
    parse_response,
)


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = build_messages(DEMO_QUESTION)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": DEMO_QUESTION})

    def test_system_prompt_blocks_diagnosis_and_asks_for_privacy(self):
        text = SYSTEM_PROMPT.lower()
        for phrase in ("diagnos", "prescrib", "clinician", "emergency", "identifying"):
            self.assertIn(phrase, text)

    def test_question_is_trimmed(self):
        messages = build_messages("  hello  ")
        self.assertEqual(messages[-1]["content"], "hello")

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        messages = build_messages("question", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_empty_question_is_rejected(self):
        for value in ("", "   ", None):
            with self.assertRaises(ValueError):
                build_messages(value)

    def test_too_long_question_is_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("a" * (MAX_QUESTION_CHARS + 1))

    def test_invalid_history_is_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("q", [{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            build_messages("q", [{"role": "user", "content": "  "}])


class BuildPayloadTest(unittest.TestCase):
    def test_payload_targets_spark(self):
        payload = build_payload(build_messages("hello"))
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertEqual(payload["max_tokens"], MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1]["content"], "hello")

    def test_empty_messages_are_rejected(self):
        with self.assertRaises(ValueError):
            build_payload([])


class ParseResponseTest(unittest.TestCase):
    def test_plain_content(self):
        data = {"choices": [{"message": {"content": "  See a clinician.  "}}]}
        self.assertEqual(parse_response(data), "See a clinician.")

    def test_content_parts_are_joined(self):
        data = {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}
        self.assertEqual(parse_response(data), "ab")

    def test_bad_shapes_are_rejected(self):
        for data in (
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {"content": "   "}}]},
            {"choices": [{"message": {"content": None}}]},
            "not an object",
        ):
            with self.assertRaises(ValueError):
                parse_response(data)


if __name__ == "__main__":
    unittest.main()
