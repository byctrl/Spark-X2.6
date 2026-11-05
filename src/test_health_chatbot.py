"""Tests for the Spark-X2.6-4B health-information chatbot.

Run from the repository root:
    python -m unittest discover -s src -p "test_*.py"
"""

import json
import os
import sys
import unittest
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc  # noqa: E402


class FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class BuildMessagesTests(unittest.TestCase):
    def test_model_id_and_budget(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)
        self.assertTrue(hc.DEMO_QUESTION.strip())

    def test_system_prompt_keeps_safety_rules(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("never prescribe", prompt)
        self.assertIn("emergency", prompt)

    def test_messages_start_with_system_and_end_with_user(self):
        messages = hc.build_messages("  demam tiga hari  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "demam tiga hari"})
        self.assertEqual(len(messages), 2)

    def test_history_is_preserved_in_order(self):
        history = [
            {"role": "user", "content": "halo"},
            {"role": "assistant", "content": "halo, ada yang bisa dibantu?"},
        ]
        messages = hc.build_messages("batuk kering", history=history)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])

    def test_invalid_question_is_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_invalid_history_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("halo", history=[{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("halo", history=[{"role": "user", "content": ""}])
        with self.assertRaises(ValueError):
            hc.build_messages("halo", history=["not-a-dict"])


class PayloadTests(unittest.TestCase):
    def test_payload_shape(self):
        payload = hc.build_payload(hc.build_messages("halo"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])

    def test_payload_requires_final_user_turn(self):
        with self.assertRaises(ValueError):
            hc.build_payload([{"role": "system", "content": "x"}])


class ExtractReplyTests(unittest.TestCase):
    def test_extracts_and_trims(self):
        body = {"choices": [{"message": {"content": "  jawaban  "}}]}
        self.assertEqual(hc.extract_reply(body), "jawaban")

    def test_rejects_bad_shapes(self):
        for body in ({}, {"choices": []}, {"choices": [{}]},
                     {"choices": [{"message": {"content": "   "}}]}):
            with self.assertRaises(RuntimeError):
                hc.extract_reply(body)


class GenerateTests(unittest.TestCase):
    def test_posts_to_chat_completions(self):
        captured = {}

        def opener(request, timeout=None):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

        reply = hc.generate(
            hc.build_messages("halo"),
            base_url="http://localhost:30000/v1/",
            timeout=7,
            opener=opener,
        )

        self.assertEqual(reply, "ok")
        self.assertEqual(captured["url"], "http://localhost:30000/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(captured["timeout"], 7)

    def test_unreachable_endpoint_is_reported(self):
        def opener(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        with self.assertRaises(RuntimeError):
            hc.generate(hc.build_messages("halo"), opener=opener)

    def test_emergency_prefix_is_detected(self):
        self.assertTrue(hc.is_emergency_reply("EMERGENCY: call 112 now"))
        self.assertFalse(hc.is_emergency_reply("General information only."))


if __name__ == "__main__":
    unittest.main()
