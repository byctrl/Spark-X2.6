"""Tests for src/health_chatbot.py (stdlib only, no model download needed)."""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class MessagesTest(unittest.TestCase):
    def test_system_prompt_is_safety_bounded(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        for phrase in ("diagnosis", "dokter", "darurat", "privasi"):
            self.assertIn(phrase, prompt)

    def test_build_messages_puts_system_first_and_question_last(self):
        messages = hc.build_messages("  Apa itu demam?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Apa itu demam?"})
        self.assertEqual(len(messages), 2)

    def test_build_messages_keeps_history(self):
        history = [
            {"role": "user", "content": "halo"},
            {"role": "assistant", "content": "hai"},
        ]
        messages = hc.build_messages("lanjut", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_build_messages_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")
        with self.assertRaises(ValueError):
            hc.build_messages("ok", [{"role": "tool", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("ok", ["not-a-dict"])


class PayloadTest(unittest.TestCase):
    def test_payload_targets_spark_and_bounds_tokens(self):
        payload = hc.build_payload("Apa itu demam?")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(len(payload["messages"]), 2)


class ExtractContentTest(unittest.TestCase):
    def test_reads_message_content(self):
        data = {"choices": [{"message": {"content": "  jawaban  "}}]}
        self.assertEqual(hc.extract_content(data), "jawaban")

    def test_rejects_malformed_or_empty(self):
        bad = (
            {},
            None,
            {"choices": []},
            {"choices": [{"message": {"content": "   "}}]},
            {"choices": [{"message": {}}]},
        )
        for data in bad:
            with self.assertRaises(hc.ChatbotError):
                hc.extract_content(data)


class ChatTest(unittest.TestCase):
    def test_chat_sends_payload_to_chat_completions(self):
        seen = {}

        def fake_requester(url, payload, timeout):
            seen["url"] = url
            seen["payload"] = payload
            return {"choices": [{"message": {"content": "Minum air cukup."}}]}

        text = hc.chat(
            "Apa itu demam?",
            base_url="http://127.0.0.1:30000/v1/",
            requester=fake_requester,
        )
        self.assertEqual(text, "Minum air cukup.")
        self.assertEqual(seen["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(seen["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(seen["payload"]["max_tokens"], hc.MAX_NEW_TOKENS)

    def test_transport_error_becomes_chatbot_error(self):
        def broken_requester(url, payload, timeout):
            raise OSError("connection refused")

        with self.assertRaises(hc.ChatbotError):
            hc.chat("Apa itu demam?", requester=broken_requester)

    def test_answer_falls_back_when_model_unavailable(self):
        def broken_requester(url, payload, timeout):
            raise OSError("connection refused")

        text = hc.answer("Apa itu demam?", requester=broken_requester)
        self.assertIn("MODE OFFLINE", text)
        self.assertIn("dokter", text.lower())

    def test_answer_offline_skips_model_call(self):
        def exploding_requester(url, payload, timeout):
            raise AssertionError("model must not be called in offline mode")

        text = hc.answer("Apa itu demam?", requester=exploding_requester, offline=True)
        self.assertIn("MODE OFFLINE", text)


class CliTest(unittest.TestCase):
    def test_offline_cli_prints_json(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = hc.main(["--offline", "--json"])
        self.assertEqual(code, 0)
        record = json.loads(buffer.getvalue())
        self.assertEqual(record["model"], hc.MODEL_ID)
        self.assertEqual(record["question"], hc.DEMO_QUESTION)
        self.assertIn("MODE OFFLINE", record["answer"])


if __name__ == "__main__":
    unittest.main()
