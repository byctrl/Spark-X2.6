"""Offline tests for the Spark-X2.6-4B health chatbot (no model required)."""

import os
import unittest
from unittest import mock

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_roles_and_stripping(self):
        messages = hc.build_messages("  How do I prevent dehydration?  ")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "How do I prevent dehydration?")

    def test_system_prompt_keeps_safety_boundaries(self):
        system = hc.build_messages("hi")[0]["content"].lower()
        self.assertIn("must not diagnose", system)
        self.assertIn("emergency", system)

    def test_blank_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")
        with self.assertRaises(ValueError):
            hc.build_messages(None)


class BuildPayloadTest(unittest.TestCase):
    def test_uses_model_constants(self):
        payload = hc.build_payload("hello")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(payload["messages"][-1]["content"], "hello")


class ExtractTextTest(unittest.TestCase):
    def test_returns_stripped_content(self):
        response = {"choices": [{"message": {"content": "  Drink water.  "}}]}
        self.assertEqual(hc.extract_text(response), "Drink water.")

    def test_missing_or_empty_content_rejected(self):
        for response in (
            {},
            {"choices": []},
            {"choices": [{"message": {"content": "   "}}]},
        ):
            with self.assertRaises(ValueError):
                hc.extract_text(response)


class ChatTest(unittest.TestCase):
    def test_calls_default_endpoint(self):
        seen = {}

        def fake_transport(url, payload, timeout):
            seen["url"] = url
            seen["payload"] = payload
            return {"choices": [{"message": {"content": "Rest and drink water."}}]}

        answer = hc.chat("I feel thirsty", transport=fake_transport)
        self.assertEqual(answer, "Rest and drink water.")
        self.assertEqual(seen["url"], hc.DEFAULT_BASE_URL + "/chat/completions")
        self.assertEqual(seen["payload"]["model"], hc.MODEL_ID)

    def test_respects_env_base_url(self):
        seen = {}

        def fake_transport(url, payload, timeout):
            seen["url"] = url
            return {"choices": [{"message": {"content": "ok"}}]}

        with mock.patch.dict(os.environ, {"SPARK_BASE_URL": "http://localhost:9999/v1/"}):
            hc.chat("q", transport=fake_transport)

        self.assertEqual(seen["url"], "http://localhost:9999/v1/chat/completions")

    def test_explicit_base_url_overrides_env(self):
        seen = {}

        def fake_transport(url, payload, timeout):
            seen["url"] = url
            return {"choices": [{"message": {"content": "ok"}}]}

        with mock.patch.dict(os.environ, {"SPARK_BASE_URL": "http://localhost:9999/v1"}):
            hc.chat("q", base_url="http://127.0.0.1:30001/v1", transport=fake_transport)

        self.assertEqual(seen["url"], "http://127.0.0.1:30001/v1/chat/completions")

    def test_empty_model_output_rejected(self):
        with self.assertRaises(ValueError):
            hc.chat(
                "q",
                transport=lambda url, payload, timeout: {
                    "choices": [{"message": {"content": "  "}}]
                },
            )


if __name__ == "__main__":
    unittest.main()
