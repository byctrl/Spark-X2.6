"""Offline tests for the Spark-X2.6-4B health chatbot demo."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("hello")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "hello"})

    def test_question_is_stripped(self):
        messages = hc.build_messages("  cough  ")
        self.assertEqual(messages[-1]["content"], "cough")

    def test_empty_question_is_rejected(self):
        for value in ("", "   ", None, 7):
            with self.assertRaises(ValueError):
                hc.build_messages(value)

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        messages = hc.build_messages("second", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "second")

    def test_invalid_history_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("q", [{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("q", ["not-a-dict"])


class BuildPayloadTest(unittest.TestCase):
    def test_payload_uses_model_and_max_tokens(self):
        payload = hc.build_payload(hc.build_messages("hi"))
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertIn("messages", payload)

    def test_bad_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_payload([])
        with self.assertRaises(ValueError):
            hc.build_payload([{"role": "user", "content": "x"}], max_tokens=0)


class ExtractAnswerTest(unittest.TestCase):
    def test_valid_response(self):
        body = {"choices": [{"message": {"content": "  see a clinician  "}}]}
        self.assertEqual(hc.extract_answer(body), "see a clinician")

    def test_malformed_response(self):
        for body in ({}, {"choices": []}, {"choices": [{}]}, None):
            with self.assertRaises(ValueError):
                hc.extract_answer(body)

    def test_empty_answer(self):
        body = {"choices": [{"message": {"content": "   "}}]}
        with self.assertRaises(ValueError):
            hc.extract_answer(body)


class DemoTest(unittest.TestCase):
    def test_demo_question_and_prompt_guardrails(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())
        prompt = hc.SYSTEM_PROMPT.lower()
        for phrase in ("not medical advice", "diagnose", "clinician"):
            self.assertIn(phrase, prompt)

    def test_unreachable_endpoint_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            hc.chat("hello", base_url="http://127.0.0.1:9/v1", timeout=1)


if __name__ == "__main__":
    unittest.main()
