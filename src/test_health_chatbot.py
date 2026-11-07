"""Unit tests for src/health_chatbot.py (no model download required).

Run from the repository root:
    python -m unittest discover -s src -p "test_*.py"
"""

import unittest

from health_chatbot import (
    DISCLAIMER,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    build_messages,
    build_payload,
    chat,
    extract_answer,
    with_disclaimer,
)


class BuildMessagesTests(unittest.TestCase):
    def test_builds_system_and_user_turns(self):
        messages = build_messages("  Demam tiga hari  ")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[1], {"role": "user", "content": "Demam tiga hari"})

    def test_rejects_empty_question(self):
        for value in ("", "   ", None):
            with self.assertRaises(ValueError):
                build_messages(value)


class BuildPayloadTests(unittest.TestCase):
    def test_payload_is_bounded_and_non_streaming(self):
        payload = build_payload(build_messages("Apa itu demam?"))
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertEqual(payload["max_tokens"], MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertIn("messages", payload)

    def test_rejects_empty_messages(self):
        with self.assertRaises(ValueError):
            build_payload([])


class ExtractAnswerTests(unittest.TestCase):
    def test_extracts_and_strips_content(self):
        response = {"choices": [{"message": {"role": "assistant", "content": "  Jaga hidrasi.  "}}]}
        self.assertEqual(extract_answer(response), "Jaga hidrasi.")

    def test_rejects_malformed_or_empty_content(self):
        for response in ({}, {"choices": []}, {"choices": [{"message": {"content": "  "}}]}, None):
            with self.assertRaises(ValueError):
                extract_answer(response)


class DisclaimerTests(unittest.TestCase):
    def test_appends_disclaimer_once(self):
        once = with_disclaimer("Jaga hidrasi.")
        self.assertTrue(once.startswith("Jaga hidrasi."))
        self.assertIn("bukan diagnosis", once)
        self.assertEqual(with_disclaimer(once), once)
        self.assertIn(DISCLAIMER.strip(), once)


class ChatTests(unittest.TestCase):
    def test_calls_openai_compatible_endpoint(self):
        calls = {}

        def fake_transport(url, payload, timeout):
            calls["url"] = url
            calls["payload"] = payload
            calls["timeout"] = timeout
            return {"choices": [{"message": {"content": "  Ke klinik bila demam menetap.  "}}]}

        answer = chat(
            "Demam pasca perjalanan",
            base_url="http://127.0.0.1:30000/v1/",
            transport=fake_transport,
            timeout=5,
        )

        self.assertEqual(calls["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(calls["payload"]["model"], MODEL_ID)
        self.assertEqual(calls["payload"]["max_tokens"], MAX_NEW_TOKENS)
        self.assertEqual(calls["timeout"], 5)
        self.assertTrue(answer.startswith("Ke klinik bila demam menetap."))
        self.assertIn("bukan diagnosis", answer)

    def test_propagates_transport_errors(self):
        def failing_transport(url, payload, timeout):
            raise ValueError("unexpected response shape from the Spark endpoint")

        with self.assertRaises(ValueError):
            chat("Demam", base_url="http://127.0.0.1:30000/v1", transport=failing_transport)

    def test_module_exposes_model_id(self):
        self.assertEqual(MODEL_ID, "XHToken/Spark-X2.6-4B")


if __name__ == "__main__":
    unittest.main()
