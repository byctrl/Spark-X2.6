"""Offline tests for src/health_chatbot.py (no model download needed)."""

import unittest

import health_chatbot as hc


class HealthChatbotTests(unittest.TestCase):
    def test_model_id(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_max_new_tokens(self):
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_has_safety_rules(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("diagnose", prompt)
        self.assertIn("prescribe", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("not medical advice", prompt)

    def test_build_messages_shape(self):
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[1], {"role": "user", "content": hc.DEMO_QUESTION})

    def test_build_messages_keeps_valid_history(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("  fever  ", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"], "hi")
        self.assertEqual(messages[2]["content"], "hello")
        self.assertEqual(messages[-1], {"role": "user", "content": "fever"})

    def test_empty_question_rejected(self):
        for bad in ("", "   ", None, 123):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_build_payload(self):
        payload = hc.build_payload(hc.build_messages(hc.DEMO_QUESTION))
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])

    def test_build_payload_rejects_empty_messages(self):
        with self.assertRaises(ValueError):
            hc.build_payload([])

    def test_build_request_url_and_headers(self):
        request = hc.build_request(hc.build_messages("cough"), base_url="http://example.test/v1/")
        self.assertEqual(request.full_url, "http://example.test/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertIsNotNone(request.get_header("Content-type"))

    def test_build_request_adds_auth_only_when_given(self):
        request = hc.build_request(hc.build_messages("cough"), base_url="http://example.test/v1")
        self.assertIsNone(request.get_header("Authorization"))
        request = hc.build_request(
            hc.build_messages("cough"), base_url="http://example.test/v1", api_key="k"
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer k")

    def test_parse_response(self):
        data = {"choices": [{"message": {"content": "  Rest and drink water.  "}}]}
        self.assertEqual(hc.parse_response(data), "Rest and drink water.")

    def test_parse_response_rejects_bad_payloads(self):
        bad_payloads = [
            {},
            {"choices": []},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": "   "}}]},
            {"choices": [{"message": {"content": None}}]},
        ]
        for bad in bad_payloads:
            with self.assertRaises(ValueError):
                hc.parse_response(bad)


if __name__ == "__main__":
    unittest.main()
