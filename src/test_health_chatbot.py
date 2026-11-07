"""Offline tests for the Spark-X2.6-4B health chatbot demo.

No model download and no network access is required.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hb


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_blocks_diagnosis(self):
        prompt = hb.SYSTEM_PROMPT.lower()
        self.assertIn("not diagnose", prompt)
        self.assertIn("clinician", prompt)

    def test_system_message_first(self):
        messages = hb.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hb.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_question_is_stripped(self):
        messages = hb.build_messages("  hello  ")
        self.assertEqual(messages[-1]["content"], "hello")

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hb.build_messages("   ")
        with self.assertRaises(ValueError):
            hb.build_messages(None)

    def test_history_is_forwarded(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = hb.build_messages("About a week.", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1], history[0])
        self.assertEqual(messages[2], history[1])

    def test_bad_history_rejected(self):
        with self.assertRaises(ValueError):
            hb.build_messages("hi", [{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hb.build_messages("hi", [{"role": "user", "content": ""}])


class BuildPayloadTest(unittest.TestCase):
    def test_uses_model_and_token_budget(self):
        payload = hb.build_payload("hello")
        self.assertEqual(payload["model"], hb.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hb.MAX_NEW_TOKENS)
        self.assertEqual(hb.MAX_NEW_TOKENS, 1024)
        self.assertFalse(payload["stream"])

    def test_custom_max_tokens(self):
        payload = hb.build_payload("hello", max_tokens=64)
        self.assertEqual(payload["max_tokens"], 64)


class ExtractAnswerTest(unittest.TestCase):
    def test_reads_content(self):
        response = {"choices": [{"message": {"content": "  See a clinician.  "}}]}
        self.assertEqual(hb.extract_answer(response), "See a clinician.")

    def test_rejects_missing_content(self):
        with self.assertRaises(ValueError):
            hb.extract_answer({})
        with self.assertRaises(ValueError):
            hb.extract_answer({"choices": []})
        with self.assertRaises(ValueError):
            hb.extract_answer({"choices": [{"message": {"content": "  "}}]})


class ChatTest(unittest.TestCase):
    def test_chat_wraps_model_answer(self):
        fake = {"choices": [{"message": {"content": "Drink water and rest."}}]}
        with mock.patch.object(hb, "_post", return_value=fake) as post:
            result = hb.chat("What helps with a mild fever?")
        self.assertEqual(result["answer"], "Drink water and rest.")
        self.assertEqual(result["disclaimer"], hb.DISCLAIMER)
        self.assertEqual(result["model"], hb.MODEL_ID)
        path, payload = post.call_args[0][0], post.call_args[0][1]
        self.assertEqual(path, "/chat/completions")
        self.assertEqual(payload["model"], hb.MODEL_ID)

    def test_chat_requires_question(self):
        with self.assertRaises(ValueError):
            hb.chat("")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
