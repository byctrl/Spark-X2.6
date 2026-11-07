"""Offline tests for health_chatbot.py (no model weights required)."""

import unittest

import health_chatbot as hb


class ValidateQuestionTests(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(hb.validate_question("  I   have\n a cough "), "I have a cough")

    def test_rejects_empty(self):
        with self.assertRaises(hb.ChatbotError):
            hb.validate_question("   ")

    def test_rejects_non_string(self):
        with self.assertRaises(hb.ChatbotError):
            hb.validate_question(None)

    def test_rejects_overlong_question(self):
        with self.assertRaises(hb.ChatbotError):
            hb.validate_question("a" * (hb.MAX_QUESTION_CHARS + 1))


class BuildPromptTests(unittest.TestCase):
    def test_messages_start_with_safety_system_prompt(self):
        messages = hb.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("not medical advice", messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_kept_between_system_and_question(self):
        history = [
            {"role": "user", "content": "  hello  "},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hb.build_messages("next", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "hello")

    def test_payload_targets_spark_model(self):
        payload = hb.build_payload("question")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])


class ExtractTextTests(unittest.TestCase):
    def test_reads_content(self):
        response = {"choices": [{"message": {"content": "  Drink water.  "}, "finish_reason": "stop"}]}
        self.assertEqual(hb.extract_text(response), "Drink water.")

    def test_reports_exhausted_budget(self):
        response = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        with self.assertRaises(hb.ChatbotError) as ctx:
            hb.extract_text(response)
        self.assertIn("length", str(ctx.exception))

    def test_rejects_malformed_response(self):
        with self.assertRaises(hb.ChatbotError):
            hb.extract_text({"choices": []})
        with self.assertRaises(hb.ChatbotError):
            hb.extract_text("not a dict")


class ChatTests(unittest.TestCase):
    def test_chat_uses_transport_and_adds_disclaimer(self):
        captured = []

        def fake_transport(payload):
            captured.append(payload)
            return {"choices": [{"message": {"content": "Rest and watch for red flags."}, "finish_reason": "stop"}]}

        answer = hb.chat("I have a cough", transport=fake_transport)

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["model"], hb.MODEL_ID)
        self.assertTrue(answer.startswith("Rest and watch for red flags."))
        self.assertIn("not medical advice", answer.lower())

    def test_disclaimer_is_not_duplicated(self):
        text = "Rest. This is not medical advice."
        self.assertEqual(hb.with_disclaimer(text), text)

    def test_chat_propagates_endpoint_failure(self):
        def failing_transport(payload):
            raise hb.ChatbotError("cannot reach endpoint")

        with self.assertRaises(hb.ChatbotError):
            hb.chat("hello", transport=failing_transport)

    def test_main_offline_prints_prompt(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = hb.main(["I have a fever"])
        self.assertEqual(code, 1)
        self.assertIn("prompt that would be sent", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
