"""Offline tests for src/health_chatbot.py (no model or network needed)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.health_chatbot import (  # noqa: E402
    DEMO_QUESTION,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    build_messages,
    build_payload,
    parse_completion,
)


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = build_messages("What is a fever?")
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})

    def test_question_is_last(self):
        messages = build_messages("What is a fever?")
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_preserved_and_not_mutated(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignored"},
            {"role": "tool", "content": "ignored"},
        ]
        snapshot = [dict(turn) for turn in history]
        messages = build_messages("next", history=history)
        self.assertEqual(history, snapshot)
        self.assertEqual(
            messages[1:-1],
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )

    def test_system_prompt_forbids_diagnosis(self):
        self.assertIn("must not diagnose", SYSTEM_PROMPT)
        self.assertIn("emergency", SYSTEM_PROMPT)


class PayloadTest(unittest.TestCase):
    def test_payload_shape(self):
        payload = build_payload(build_messages(DEMO_QUESTION))
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertEqual(payload["max_tokens"], MAX_NEW_TOKENS)
        self.assertIsInstance(payload["messages"], list)
        self.assertEqual(payload["messages"][-1]["content"], DEMO_QUESTION)


class ParseCompletionTest(unittest.TestCase):
    def test_extracts_and_strips_content(self):
        data = {"choices": [{"message": {"role": "assistant", "content": " hi "}}]}
        self.assertEqual(parse_completion(data), "hi")

    def test_rejects_malformed_payload(self):
        for bad in ({}, {"choices": []}, {"choices": [{"message": {}}]}, None):
            with self.assertRaises(ValueError):
                parse_completion(bad)

    def test_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            parse_completion({"choices": [{"message": {"content": "   "}}]})


if __name__ == "__main__":
    unittest.main()
