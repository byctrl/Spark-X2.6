"""Offline tests for src/health_chatbot.py (no model download required)."""

import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_single_turn_shape(self):
        messages = hc.build_messages("  What is hydration?  ")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is hydration?"})

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
        ]
        messages = hc.build_messages("Tell me more", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "Tell me more")

    def test_empty_question_rejected(self):
        for value in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(value)

    def test_bad_history_rejected(self):
        for history in ([{"role": "system", "content": "x"}], [{"role": "user", "content": " "}]):
            with self.assertRaises(ValueError):
                hc.build_messages("q", history)

    def test_prompt_refuses_diagnosis(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("clinician", prompt)


class BuildPayloadTest(unittest.TestCase):
    def test_defaults(self):
        payload = hc.build_payload(hc.build_messages("q"))
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(payload["messages"][-1]["content"], "q")


class ParseCompletionTest(unittest.TestCase):
    def test_valid(self):
        data = {"choices": [{"message": {"content": "  Drink water.  "}}]}
        self.assertEqual(hc.parse_completion(data), "Drink water.")

    def test_broken_payloads(self):
        for data in ({}, {"choices": []}, {"choices": [{"message": {"content": "   "}}]}, None):
            with self.assertRaises(ValueError):
                hc.parse_completion(data)


class DisclaimerTest(unittest.TestCase):
    def test_appended_once(self):
        once = hc.ensure_disclaimer("Rest and fluids.")
        self.assertTrue(once.startswith("Rest and fluids."))
        self.assertIn(hc.DISCLAIMER, once)
        self.assertEqual(hc.ensure_disclaimer(once), once)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            hc.ensure_disclaimer("   ")


class DemoTest(unittest.TestCase):
    def test_demo_question_is_usable(self):
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertEqual(messages[-1]["content"], hc.DEMO_QUESTION)


if __name__ == "__main__":
    unittest.main()
