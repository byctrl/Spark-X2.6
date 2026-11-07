"""Offline unit tests for src/health_chatbot.py (no model, no network)."""

import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_returns_system_then_user(self):
        messages = hc.build_messages("  I have a headache  ")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "I have a headache")

    def test_rejects_empty_question(self):
        for value in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(value)

    def test_system_prompt_is_safety_bounded(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("clinician", prompt)
        self.assertTrue(hc.DISCLAIMER)
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)
        self.assertTrue(hc.DEMO_QUESTION.strip())


class BuildPayloadTest(unittest.TestCase):
    def test_payload_shape(self):
        payload = hc.build_payload("How do I lower fever?")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1]["role"], "user")

    def test_payload_rejects_bad_max_tokens(self):
        with self.assertRaises(ValueError):
            hc.build_payload("question", max_tokens=0)


class ExtractAnswerTest(unittest.TestCase):
    def test_reads_message_content(self):
        body = {"choices": [{"message": {"content": "  Drink water.  "}}]}
        self.assertEqual(hc.extract_answer(body), "Drink water.")

    def test_rejects_empty_choices(self):
        with self.assertRaises(ValueError):
            hc.extract_answer({"choices": []})

    def test_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            hc.extract_answer({"choices": [{"message": {"content": "   "}}]})


if __name__ == "__main__":
    unittest.main()
