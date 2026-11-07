#!/usr/bin/env python3
"""Small offline tests for health_chatbot.py (no model server needed)."""

import unittest

import health_chatbot as hb


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hb.build_messages("  Apa itu demam?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hb.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Apa itu demam?"})

    def test_history_is_kept_between_system_and_question(self):
        history = [
            {"role": "user", "content": "Saya demam."},
            {"role": "assistant", "content": "Sejak bila?"},
            {"role": "tool", "content": "ignored"},
        ]
        messages = hb.build_messages("Tiga hari.", history)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hb.build_messages("   ")


class SafetyTest(unittest.TestCase):
    def test_emergency_phrase_detected(self):
        self.assertTrue(hb.needs_emergency_notice("I have chest pain since morning"))
        self.assertTrue(hb.needs_emergency_notice("sesak nafas dan pendarahan"))

    def test_ordinary_question_not_flagged(self):
        self.assertFalse(hb.needs_emergency_notice(hb.DEMO_QUESTION))

    def test_notice_prepended_for_emergency(self):
        answer = hb.ask_health_question("chest pain", chat_fn=lambda m: "See a clinician.")
        self.assertTrue(answer.startswith(hb.EMERGENCY_NOTICE))
        self.assertIn("See a clinician.", answer)

    def test_notice_not_added_otherwise(self):
        answer = hb.ask_health_question("common cold", chat_fn=lambda m: "Rest and fluids.")
        self.assertEqual(answer, "Rest and fluids.")


class ChatRequestTest(unittest.TestCase):
    def test_chat_posts_openai_payload(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"choices": [{"message": {"content": " ok "}}]}'

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["body"] = request.data
            captured["timeout"] = timeout
            return FakeResponse()

        original = hb.urllib.request.urlopen
        hb.urllib.request.urlopen = fake_urlopen
        try:
            reply = hb.chat(hb.build_messages("hello"), base_url="http://127.0.0.1:30000/v1")
        finally:
            hb.urllib.request.urlopen = original

        self.assertEqual(reply, "ok")
        self.assertEqual(captured["url"], "http://127.0.0.1:30000/v1/chat/completions")
        import json
        payload = json.loads(captured["body"].decode("utf-8"))
        self.assertEqual(payload["model"], hb.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hb.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])


if __name__ == "__main__":
    unittest.main()
