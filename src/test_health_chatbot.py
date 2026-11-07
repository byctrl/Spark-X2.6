"""Tests for src/health_chatbot.py (no network, no GPU, no model weights)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as chatbot


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = chatbot.build_messages("  I have a fever.  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Do not diagnose", messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "I have a fever."})
        self.assertEqual(len(messages), 2)

    def test_keeps_history_between_system_and_question(self):
        history = [
            {"role": "user", "content": "What is a fever?"},
            {"role": "assistant", "content": "A fever is a higher body temperature."},
        ]
        messages = chatbot.build_messages("And a cough?", history)
        self.assertEqual(
            [m["role"] for m in messages], ["system", "user", "assistant", "user"]
        )

    def test_rejects_empty_question(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                chatbot.build_messages(bad)

    def test_rejects_bad_history(self):
        with self.assertRaises(ValueError):
            chatbot.build_messages("hi", [{"role": "system", "content": "x"}])


class PayloadTest(unittest.TestCase):
    def test_payload_uses_model_and_token_limit(self):
        payload = chatbot.build_payload(chatbot.build_messages("hi"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], chatbot.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])

    def test_rejects_empty_messages(self):
        with self.assertRaises(ValueError):
            chatbot.build_payload([])


class ExtractAnswerTest(unittest.TestCase):
    def test_reads_and_strips_content(self):
        data = {"choices": [{"message": {"content": "  Drink water.  "}}]}
        self.assertEqual(chatbot.extract_answer(data), "Drink water.")

    def test_rejects_empty_or_bad_shape(self):
        bad_payloads = (
            {},
            {"choices": []},
            {"choices": [{"message": {"content": "   "}}]},
        )
        for data in bad_payloads:
            with self.assertRaises(ValueError):
                chatbot.extract_answer(data)


class ChatTest(unittest.TestCase):
    def test_uses_injected_transport(self):
        calls = []

        def fake_transport(url, payload, timeout):
            calls.append((url, payload, timeout))
            return {"choices": [{"message": {"content": "Rest and watch for warning signs."}}]}

        reply = chatbot.ask(
            "I have a cough.",
            base_url="http://127.0.0.1:9/v1/",
            timeout=5,
            transport=fake_transport,
        )
        self.assertTrue(reply.startswith("Rest and watch for warning signs."))
        self.assertIn("not medical advice", reply)
        self.assertEqual(calls[0][0], "http://127.0.0.1:9/v1/chat/completions")
        self.assertEqual(calls[0][1]["model"], chatbot.MODEL_ID)
        self.assertEqual(calls[0][2], 5)

    def test_transport_error_is_not_hidden(self):
        def failing_transport(url, payload, timeout):
            raise OSError("connection refused")

        with self.assertRaises(OSError):
            chatbot.ask("hi", transport=failing_transport)


class DemoTest(unittest.TestCase):
    def test_demo_question_points_to_a_clinician(self):
        self.assertTrue(chatbot.DEMO_QUESTION.strip())
        self.assertIn("clinician", chatbot.DEMO_QUESTION.lower())

    def test_disclaimer_is_attached(self):
        reply = chatbot.format_reply("General information only.")
        self.assertIn("not medical advice", reply)


if __name__ == "__main__":
    unittest.main()
