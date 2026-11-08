"""Offline tests for the Spark-X2.6-4B health chatbot (no network, no model)."""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def reply_payload(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": hc.DEMO_QUESTION})

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_is_kept_and_unknown_roles_are_dropped(self):
        history = [
            {"role": "user", "content": "I live far from a clinic."},
            {"role": "assistant", "content": "Noted."},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("What next?", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )


class ParseReplyTest(unittest.TestCase):
    def test_extracts_text(self):
        self.assertEqual(hc.parse_reply(reply_payload("  See a clinician.  ")), "See a clinician.")

    def test_rejects_malformed_payload(self):
        with self.assertRaises(ValueError):
            hc.parse_reply({"choices": []})

    def test_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            hc.parse_reply(reply_payload("   "))


class AnswerTest(unittest.TestCase):
    def test_uses_injected_transport(self):
        seen = {}

        def transport(messages):
            seen["messages"] = messages
            return reply_payload("Ask the clinic about the cough and the travel history.")

        reply = hc.answer("I have a cough.", transport=transport)
        self.assertIn("travel history", reply)
        self.assertNotIn(hc.EMERGENCY_NOTICE, reply)
        self.assertEqual(seen["messages"][0]["role"], "system")

    def test_emergency_notice_is_prepended_locally(self):
        def transport(messages):
            return reply_payload("Book a routine appointment.")

        reply = hc.answer("I have chest pain since this morning.", transport=transport)
        self.assertTrue(reply.startswith(hc.EMERGENCY_NOTICE))

    def test_emergency_notice_is_not_duplicated(self):
        def transport(messages):
            return reply_payload(hc.EMERGENCY_NOTICE + "\n\nCall now.")

        reply = hc.answer("I cannot breathe.", transport=transport)
        self.assertEqual(reply.count(hc.EMERGENCY_NOTICE), 1)


class PostChatTest(unittest.TestCase):
    def test_request_shape_and_endpoint(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse(reply_payload("ok"))

        messages = hc.build_messages("Question?")
        with mock.patch.object(hc.urllib.request, "urlopen", side_effect=fake_urlopen):
            payload = hc.post_chat(messages, base_url="http://127.0.0.1:9/v1/")

        self.assertEqual(hc.parse_reply(payload), "ok")
        self.assertEqual(captured["url"], "http://127.0.0.1:9/v1/chat/completions")
        self.assertEqual(captured["body"]["model"], hc.MODEL_ID)
        self.assertEqual(captured["body"]["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(captured["body"]["stream"])
        self.assertEqual(captured["body"]["messages"], messages)


if __name__ == "__main__":
    unittest.main()
