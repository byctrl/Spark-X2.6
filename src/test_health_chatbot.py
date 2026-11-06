"""Offline tests for src/health_chatbot.py (no model server required)."""

import unittest

import health_chatbot as hc


class CheckQuestionTests(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(hc.check_question("  fever\n\n and   cough "), "fever and cough")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            hc.check_question("   ")

    def test_rejects_overlong(self):
        with self.assertRaises(ValueError):
            hc.check_question("a" * (hc.MAX_QUESTION_CHARS + 1))

    def test_rejects_non_string(self):
        with self.assertRaises(TypeError):
            hc.check_question(None)


class EmergencyNoticeTests(unittest.TestCase):
    def test_detects_keyword(self):
        self.assertTrue(hc.needs_emergency_notice("I have chest pain right now"))

    def test_ignores_ordinary_text(self):
        self.assertFalse(hc.needs_emergency_notice("How much water should I drink?"))

    def test_banner_is_added(self):
        answer = hc.format_answer("chest pain", "Please seek help.")
        self.assertIn(hc.EMERGENCY_NOTICE, answer)
        self.assertIn("not medical advice", answer)


class BuildMessagesTests(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_kept_and_filtered(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
            "not-a-dict",
        ]
        messages = hc.build_messages("What is a fever?", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_payload_uses_published_model(self):
        payload = hc.build_payload("What is a fever?")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertIn("diagnose", hc.SYSTEM_PROMPT)


class ExtractAnswerTests(unittest.TestCase):
    def test_reads_message_content(self):
        response = {"choices": [{"message": {"content": "  Rest and fluids.  "}}]}
        self.assertEqual(hc.extract_answer(response), "Rest and fluids.")

    def test_rejects_empty_content(self):
        with self.assertRaises(hc.SparkRequestError):
            hc.extract_answer({"choices": [{"message": {"content": "  "}}]})

    def test_rejects_bad_shape(self):
        with self.assertRaises(hc.SparkRequestError):
            hc.extract_answer({"error": "boom"})


if __name__ == "__main__":
    unittest.main()
