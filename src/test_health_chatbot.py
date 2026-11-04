"""Offline tests for src/health_chatbot.py (no network, no model weights)."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_and_question(self):
        messages = hc.build_messages("I have a cough.")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "I have a cough."})
        self.assertEqual(len(messages), 2)

    def test_history_kept_between_system_and_question(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        messages = hc.build_messages("next", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1:3], history)

    def test_blank_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class PayloadTest(unittest.TestCase):
    def test_payload_targets_spark_model(self):
        payload = hc.build_payload(hc.build_messages("hello"))
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])


class ParseResponseTest(unittest.TestCase):
    def test_answer_gets_safety_note(self):
        data = {"choices": [{"message": {"content": "Rest and drink fluids."}}]}
        answer = hc.parse_response(data)
        self.assertTrue(answer.startswith("Rest and drink fluids."))
        self.assertIn(hc.SAFETY_NOTE, answer)

    def test_safety_note_not_duplicated(self):
        data = {"choices": [{"message": {"content": "See a clinician.\n\n" + hc.SAFETY_NOTE}}]}
        self.assertEqual(hc.parse_response(data).count(hc.SAFETY_NOTE), 1)

    def test_empty_answer_rejected(self):
        with self.assertRaises(ValueError):
            hc.parse_response({"choices": [{"message": {"content": "  "}}]})

    def test_malformed_response_rejected(self):
        with self.assertRaises(ValueError):
            hc.parse_response({"error": "boom"})


class AskTest(unittest.TestCase):
    def test_ask_posts_to_endpoint_and_returns_answer(self):
        captured = {}

        def fake_transport(url, body, timeout):
            captured["url"] = url
            captured["payload"] = json.loads(body.decode("utf-8"))
            captured["timeout"] = timeout
            return json.dumps(
                {"choices": [{"message": {"content": "Rest and drink fluids."}}]}
            ).encode("utf-8")

        answer = hc.ask(
            "cough", base_url="http://localhost:9/v1/", transport=fake_transport
        )

        self.assertEqual(captured["url"], "http://localhost:9/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(captured["payload"]["messages"][0]["role"], "system")
        self.assertIn("Rest and drink fluids.", answer)
        self.assertIn(hc.SAFETY_NOTE, answer)

    def test_invalid_json_from_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            hc.ask("cough", base_url="http://localhost:9/v1", transport=lambda u, b, t: b"not json")

    def test_endpoint_error_propagates(self):
        def broken_transport(url, body, timeout):
            raise RuntimeError("connection refused")

        with self.assertRaises(RuntimeError):
            hc.ask("cough", base_url="http://localhost:9/v1", transport=broken_transport)


if __name__ == "__main__":
    unittest.main()
