#!/usr/bin/env python3
"""Small offline tests for src/health_chatbot.py (no model download needed)."""

import unittest

import health_chatbot as hc


class FakeTransport:
    def __init__(self, text="Jaga hidrasi, istirahat cukup, dan pantau suhu tubuh."):
        self.text = text
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append({"url": url, "payload": payload, "timeout": timeout})
        return {"choices": [{"message": {"role": "assistant", "content": self.text}}]}


class ConstantsTests(unittest.TestCase):
    def test_model_id(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_token_budget(self):
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_forbids_diagnosis(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", lowered)
        self.assertIn("not medical advice", lowered)


class BuildMessagesTests(unittest.TestCase):
    def test_system_prompt_is_first(self):
        messages = hc.build_messages("Apa itu demam?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Apa itu demam?"})

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_kept_in_order(self):
        history = [
            {"role": "user", "content": "Saya demam."},
            {"role": "assistant", "content": "Sejak kapan?"},
            {"role": "system", "content": "ignored"},
            {"role": "tool", "content": "ignored"},
        ]
        messages = hc.build_messages("Sejak kemarin.", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"], "Saya demam.")
        self.assertEqual(messages[2]["content"], "Sejak kapan?")


class GenerateTests(unittest.TestCase):
    def test_payload_uses_model_and_endpoint(self):
        transport = FakeTransport()
        text = hc.generate(hc.build_messages("Halo"), base_url="http://127.0.0.1:30000/v1/", transport=transport)
        self.assertEqual(text, transport.text)
        call = transport.calls[0]
        self.assertEqual(call["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(call["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(call["payload"]["max_tokens"], hc.MAX_NEW_TOKENS)

    def test_token_budget_is_capped(self):
        transport = FakeTransport()
        hc.generate(hc.build_messages("Halo"), max_tokens=8192, transport=transport)
        self.assertEqual(transport.calls[0]["payload"]["max_tokens"], hc.MAX_NEW_TOKENS)

    def test_bad_response_raises(self):
        with self.assertRaises(RuntimeError):
            hc.generate(hc.build_messages("Halo"), transport=lambda url, payload, timeout: {})

    def test_empty_answer_raises(self):
        transport = FakeTransport(text="   ")
        with self.assertRaises(RuntimeError):
            hc.generate(hc.build_messages("Halo"), transport=transport)


class AnswerTests(unittest.TestCase):
    def test_disclaimer_is_appended(self):
        transport = FakeTransport()
        result = hc.answer("Saya batuk kering tiga hari.", transport=transport)
        self.assertIn(transport.text, result)
        self.assertIn(hc.DISCLAIMER, result)
        self.assertNotIn(hc.EMERGENCY_NOTE, result)

    def test_emergency_signs_add_warning(self):
        transport = FakeTransport()
        result = hc.answer("Saya nyeri dada sejak pagi.", transport=transport)
        self.assertIn(hc.EMERGENCY_NOTE, result)
        self.assertIn(hc.DISCLAIMER, result)

    def test_demo_question_runs(self):
        transport = FakeTransport()
        result = hc.answer(hc.DEMO_QUESTION, transport=transport)
        self.assertTrue(result.startswith(transport.text))


if __name__ == "__main__":
    unittest.main(verbosity=2)
