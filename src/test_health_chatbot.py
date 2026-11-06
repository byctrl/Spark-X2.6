"""Tests for the minimal health chatbot demo (no network access required)."""

import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1], {"role": "user", "content": "What is a fever?"})

    def test_question_is_stripped(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[1]["content"], "What is a fever?")

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_prompt_keeps_safety_boundaries(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        for phrase in ("never diagnose", "never prescribe", "clinician"):
            self.assertIn(phrase, lowered)

    def test_demo_question_is_present(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


class BuildPayloadTest(unittest.TestCase):
    def test_payload_shape(self):
        payload = hc.build_payload("What is a fever?")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(len(payload["messages"]), 2)


class ChatTest(unittest.TestCase):
    def test_chat_uses_openai_compatible_url_and_returns_answer(self):
        calls = []

        def fake_transport(url, payload):
            calls.append((url, payload))
            return {"choices": [{"message": {"content": "  A fever is ...  "}}]}

        answer = hc.chat("What is a fever?", base_url="http://localhost:9/v1/", transport=fake_transport)

        self.assertEqual(answer, "A fever is ...")
        self.assertEqual(calls[0][0], "http://localhost:9/v1/chat/completions")
        self.assertEqual(calls[0][1]["model"], hc.MODEL_ID)

    def test_chat_rejects_empty_answer(self):
        def fake_transport(url, payload):
            return {"choices": [{"message": {"content": "   "}}]}

        with self.assertRaises(ValueError):
            hc.chat("What is a fever?", transport=fake_transport)

    def test_extract_answer_accepts_plain_text(self):
        self.assertEqual(hc.extract_answer("  hello  "), "hello")

    def test_extract_answer_rejects_unknown_shape(self):
        with self.assertRaises(ValueError):
            hc.extract_answer({"error": "boom"})


if __name__ == "__main__":
    unittest.main()
