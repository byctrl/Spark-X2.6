"""Offline tests for src/health_chatbot.py (no network, no model weights)."""

import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})
        self.assertEqual(len(messages), 2)

    def test_history_is_preserved_before_question(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        messages = hc.build_messages("next", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_invalid_history_role_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("q", [{"role": "system", "content": "x"}])


class PayloadTest(unittest.TestCase):
    def test_model_and_token_budget(self):
        payload = hc.build_payload(hc.build_messages("q"))
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])

    def test_empty_messages_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_payload([])


class ParseResponseTest(unittest.TestCase):
    def test_reads_assistant_content(self):
        data = {"choices": [{"message": {"role": "assistant", "content": " rest "}}]}
        self.assertEqual(hc.parse_response(data), "rest")

    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            hc.parse_response({"error": "boom"})

    def test_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            hc.parse_response({"choices": [{"message": {"content": "  "}}]})


class DisclaimerTest(unittest.TestCase):
    def test_appended_once(self):
        once = hc.with_disclaimer("answer")
        self.assertIn(hc.DISCLAIMER, once)
        self.assertTrue(once.startswith("answer"))
        self.assertEqual(hc.with_disclaimer(once), once)


class EndpointTest(unittest.TestCase):
    def test_base_url_override(self):
        self.assertEqual(
            hc.endpoint_url("http://host:1/v1/"),
            "http://host:1/v1/chat/completions",
        )

    def test_default_endpoint_is_local(self):
        self.assertTrue(hc.endpoint_url().startswith("http://127.0.0.1:30000/v1"))


if __name__ == "__main__":
    unittest.main()
