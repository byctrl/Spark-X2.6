"""Offline tests for the Spark-X2.6-4B health chatbot demo (no network needed)."""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)

    def test_user_question_is_passed_through(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[1], {"role": "user", "content": "What is a fever?"})

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_system_prompt_sets_safety_boundaries(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("not diagnose", prompt)
        self.assertIn("not prescribe", prompt)
        self.assertIn("clinician", prompt)


class BuildPayloadTest(unittest.TestCase):
    def test_payload_uses_spark_model_and_token_limit(self):
        payload = hc.build_payload("What is a fever?")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["messages"], hc.build_messages("What is a fever?"))

    def test_demo_question_is_non_empty(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


if __name__ == "__main__":
    unittest.main()
