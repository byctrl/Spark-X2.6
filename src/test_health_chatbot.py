"""Tests for the Spark-X2.6-4B health-information chatbot demo."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_includes_system_prompt_and_question(self) -> None:
        messages = hc.build_messages("what is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Never diagnose", messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "what is a fever?"})

    def test_rejects_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_is_bounded_and_filtered(self) -> None:
        history = [{"role": "user", "content": "turn %d" % i} for i in range(20)]
        history.append({"role": "tool", "content": "ignored"})
        messages = hc.build_messages("next", history)
        self.assertEqual(len(messages), 1 + hc.MAX_HISTORY_TURNS + 1)
        self.assertTrue(all(m["role"] in ("system", "user", "assistant") for m in messages))


class PayloadTest(unittest.TestCase):
    def test_payload_is_bounded_and_uses_model_id(self) -> None:
        payload = hc.build_payload(hc.build_messages("hello"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])

    def test_payload_rejects_empty_messages(self) -> None:
        with self.assertRaises(ValueError):
            hc.build_payload([])


class ParseResponseTest(unittest.TestCase):
    def test_extracts_content(self) -> None:
        data = {"choices": [{"message": {"content": "  rest and fluids  "}}]}
        self.assertEqual(hc.parse_response(data), "rest and fluids")

    def test_rejects_empty_content(self) -> None:
        with self.assertRaises(ValueError):
            hc.parse_response({"choices": [{"message": {"content": ""}}]})

    def test_rejects_bad_shape(self) -> None:
        with self.assertRaises(ValueError):
            hc.parse_response({"error": "nope"})


class SafetyTest(unittest.TestCase):
    def test_emergency_wording_is_detected(self) -> None:
        self.assertTrue(hc.looks_like_emergency("I have CHEST PAIN right now"))
        self.assertFalse(hc.looks_like_emergency("what foods help with iron?"))

    def test_emergency_bypasses_model(self) -> None:
        calls = []

        def fake_transport(url, payload, timeout):
            calls.append(url)
            return {"choices": [{"message": {"content": "model text"}}]}

        result = hc.HealthChatbot(transport=fake_transport).chat("my father is unconscious")
        self.assertEqual(result["source"], "safety_escalation")
        self.assertEqual(result["model"], None)
        self.assertEqual(calls, [])
        self.assertIn("emergency", result["answer"].lower())


class ChatTest(unittest.TestCase):
    def test_model_answer_path(self) -> None:
        def fake_transport(url, payload, timeout):
            self.assertTrue(url.endswith("/v1/chat/completions"))
            self.assertEqual(payload["model"], hc.MODEL_ID)
            return {"choices": [{"message": {"content": "See a clinician if it lasts."}}]}

        result = hc.HealthChatbot(transport=fake_transport).chat("what is a fever?")
        self.assertEqual(result["source"], "spark-x2.6-4b")
        self.assertEqual(result["answer"], "See a clinician if it lasts.")
        self.assertEqual(result["disclaimer"], hc.DISCLAIMER)

    def test_offline_fallback_is_explicit(self) -> None:
        def broken_transport(url, payload, timeout):
            raise OSError("connection refused")

        result = hc.HealthChatbot(transport=broken_transport).chat("what is a fever?")
        self.assertEqual(result["source"], "offline_fallback")
        self.assertIn(hc.DISCLAIMER, result["answer"])
        self.assertIn("what is a fever?", result["answer"])

    def test_malformed_response_falls_back(self) -> None:
        def odd_transport(url, payload, timeout):
            return {"unexpected": True}

        result = hc.HealthChatbot(transport=odd_transport).chat("what is a fever?")
        self.assertEqual(result["source"], "offline_fallback")


if __name__ == "__main__":
    unittest.main()
