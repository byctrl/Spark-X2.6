"""Offline tests for health_chatbot.py (no network, no model download)."""

import json
import unittest

from health_chatbot import (
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    build_messages,
    chat,
    extract_reply,
)


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first_and_question_last(self):
        messages = build_messages("  I have a cough  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "I have a cough"})

    def test_history_is_kept_and_junk_dropped(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
            "not-a-dict",
        ]
        messages = build_messages("next question", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])


class ChatTest(unittest.TestCase):
    def test_request_payload_and_returned_reply(self):
        captured = {}

        def transport(url, data, timeout):
            captured["url"] = url
            captured["payload"] = json.loads(data.decode("utf-8"))
            captured["timeout"] = timeout
            return {"choices": [{"message": {"role": "assistant", "content": "  See a clinician.  "}}]}

        reply = chat(
            "Fever after travel",
            base_url="http://127.0.0.1:30000/v1/",
            timeout=5,
            transport=transport,
        )

        self.assertEqual(reply, "See a clinician.")
        self.assertEqual(captured["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(captured["timeout"], 5)
        self.assertEqual(captured["payload"]["model"], MODEL_ID)
        self.assertEqual(captured["payload"]["max_tokens"], MAX_NEW_TOKENS)
        self.assertEqual(captured["payload"]["messages"][-1]["content"], "Fever after travel")

    def test_empty_question_is_rejected_before_any_request(self):
        def transport(url, data, timeout):
            raise AssertionError("transport must not be called")

        with self.assertRaises(ValueError):
            chat("   ", transport=transport)

    def test_malformed_response_raises(self):
        with self.assertRaises(RuntimeError):
            extract_reply({"choices": []})

    def test_empty_content_raises(self):
        with self.assertRaises(RuntimeError):
            extract_reply({"choices": [{"message": {"content": "   "}}]})


if __name__ == "__main__":
    unittest.main()
