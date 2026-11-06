"""Tests for the minimal Spark-X2.6-4B health chatbot.

No model server or third-party package is required: only prompt construction,
request shape and response parsing are exercised.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as chatbot


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = chatbot.build_messages("Apa itu demam?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Never diagnose", messages[0]["content"])
        self.assertEqual(
            messages[-1], {"role": "user", "content": "Apa itu demam?"}
        )

    def test_question_is_stripped(self):
        messages = chatbot.build_messages("   demam   ")
        self.assertEqual(messages[-1]["content"], "demam")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            chatbot.build_messages("   ")

    def test_history_is_kept_and_junk_is_dropped(self):
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "ignored"},
            {"role": "assistant", "content": "   "},
        ]
        messages = chatbot.build_messages("q2", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[-1]["content"], "q2")


class PayloadTest(unittest.TestCase):
    def test_model_id_and_token_budget(self):
        self.assertEqual(chatbot.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(chatbot.MAX_NEW_TOKENS, 1024)
        payload = chatbot.build_payload("demam", max_new_tokens=64)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 64)

    def test_endpoint_url_defaults_and_override(self):
        self.assertTrue(chatbot.endpoint_url("http://host:8000/v1/").endswith("/chat/completions"))
        self.assertTrue(chatbot.endpoint_url("http://host:8000/v1").startswith("http://host:8000/v1"))


class ExtractReplyTest(unittest.TestCase):
    def test_reads_and_strips_content(self):
        data = {"choices": [{"message": {"content": "  Minum air putih.  "}}]}
        self.assertEqual(chatbot.extract_reply(data), "Minum air putih.")

    def test_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            chatbot.extract_reply({"choices": [{"message": {"content": "   "}}]})

    def test_rejects_malformed_response(self):
        with self.assertRaises(ValueError):
            chatbot.extract_reply({"error": "boom"})


if __name__ == "__main__":
    unittest.main()
