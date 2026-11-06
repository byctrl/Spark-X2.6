"""Offline unit tests for src/health_chatbot.py (no model, no network)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc  # noqa: E402


class MessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("I have a cough")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "I have a cough"})
        self.assertEqual(len(messages), 2)

    def test_history_kept_in_order_and_system_turns_dropped(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("next", history)
        self.assertEqual(
            [m["role"] for m in messages], ["system", "user", "assistant", "user"]
        )

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class SafetyPromptTest(unittest.TestCase):
    def test_prompt_blocks_diagnosis_and_points_to_clinician(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("never prescribe", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("emergency", prompt)

    def test_demo_question_and_disclaimer_present(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())
        self.assertIn("not medical advice", hc.DISCLAIMER)


class PayloadTest(unittest.TestCase):
    def test_payload_uses_model_and_token_budget(self):
        payload = hc.build_payload(hc.build_messages("q"))
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])


class ExtractReplyTest(unittest.TestCase):
    def test_content_is_stripped(self):
        data = {"choices": [{"message": {"content": "  Drink water.  "}}]}
        self.assertEqual(hc.extract_reply(data), "Drink water.")

    def test_bad_shapes_raise(self):
        bad = [
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {"content": "   "}}]},
        ]
        for data in bad:
            with self.assertRaises(ValueError):
                hc.extract_reply(data)


class ChatTest(unittest.TestCase):
    def test_calls_endpoint_with_model_and_returns_reply(self):
        calls = {}

        def fake_transport(url, payload):
            calls["url"] = url
            calls["payload"] = payload
            return {"choices": [{"message": {"content": "See a clinician."}}]}

        reply = hc.chat(
            "fever for three days",
            base_url="http://localhost:30000/v1/",
            transport=fake_transport,
        )
        self.assertEqual(reply, "See a clinician.")
        self.assertEqual(calls["url"], "http://localhost:30000/v1/chat/completions")
        self.assertEqual(calls["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(calls["payload"]["messages"][-1]["content"], "fever for three days")

    def test_default_base_url_used_when_none_given(self):
        seen = {}

        def fake_transport(url, payload):
            seen["url"] = url
            return {"choices": [{"message": {"content": "ok"}}]}

        old = os.environ.pop("SPARK_BASE_URL", None)
        try:
            hc.chat("q", transport=fake_transport)
        finally:
            if old is not None:
                os.environ["SPARK_BASE_URL"] = old
        self.assertEqual(seen["url"], hc.DEFAULT_BASE_URL + "/chat/completions")

    def test_transport_error_wrapped_as_runtime_error(self):
        def boom(url, payload):
            raise OSError("connection refused")

        with self.assertRaises(RuntimeError):
            hc.chat("fever", transport=boom)

    def test_empty_question_fails_before_transport(self):
        def should_not_run(url, payload):
            raise AssertionError("transport must not be called")

        with self.assertRaises(ValueError):
            hc.chat("  ", transport=should_not_run)


if __name__ == "__main__":
    unittest.main()
