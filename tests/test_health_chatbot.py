"""Offline tests for the Spark-X2.6-4B health chatbot demo (no network)."""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

import health_chatbot as hc  # noqa: E402


class RecordingTransport:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or {
            "choices": [{"message": {"content": "Rest, drink water, see a doctor if it worsens."}}]
        }

    def __call__(self, url, payload, headers, timeout):
        self.calls.append((url, payload, headers, timeout))
        return self.response


class BuildMessagesTests(unittest.TestCase):
    def test_system_prompt_and_question_are_included(self):
        messages = hc.build_messages("  how do I treat a mild fever?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Do not diagnose", messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "how do I treat a mild fever?"})

    def test_history_is_kept_and_invalid_turns_are_dropped(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "content": "ignored"},
            {"role": "user"},
        ]
        messages = hc.build_messages("next question", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(hc.ChatbotError):
                hc.build_messages(bad)


class ResponseTests(unittest.TestCase):
    def test_extract_text(self):
        data = {"choices": [{"message": {"content": "  plain answer  "}}]}
        self.assertEqual(hc.extract_text(data), "plain answer")

    def test_extract_text_rejects_bad_shapes(self):
        for bad in ({}, {"choices": []}, {"choices": [{"message": {"content": "  "}}]}):
            with self.assertRaises(hc.ChatbotError):
                hc.extract_text(bad)

    def test_disclaimer_added_once(self):
        once = hc.with_disclaimer("answer")
        self.assertIn(hc.DISCLAIMER, once)
        self.assertEqual(hc.with_disclaimer(once), once)


class HealthChatbotTests(unittest.TestCase):
    def test_request_targets_local_spark_endpoint(self):
        transport = RecordingTransport()
        bot = hc.HealthChatbot(base_url="http://127.0.0.1:30000/v1/", transport=transport)
        bot.answer("what is a fever?")

        url, payload, headers, timeout = transport.calls[0]
        self.assertEqual(url, "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(timeout, hc.REQUEST_TIMEOUT)

    def test_answer_includes_disclaimer(self):
        bot = hc.HealthChatbot(transport=RecordingTransport())
        answer = bot.answer("I have a cough")
        self.assertIn("Rest, drink water", answer)
        self.assertTrue(answer.endswith(hc.DISCLAIMER))

    def test_api_key_is_forwarded_when_given(self):
        transport = RecordingTransport()
        bot = hc.HealthChatbot(api_key="secret", transport=transport)
        bot.answer("hello")
        self.assertEqual(transport.calls[0][2]["Authorization"], "Bearer secret")

    def test_transport_errors_propagate_as_chatbot_error(self):
        def boom(url, payload, headers, timeout):
            raise hc.ChatbotError("offline")

        bot = hc.HealthChatbot(transport=boom)
        with self.assertRaises(hc.ChatbotError):
            bot.answer("hello")

    def test_main_reports_error_without_network(self):
        original = hc.DEFAULT_BASE_URL
        hc.DEFAULT_BASE_URL = "http://127.0.0.1:9/v1"
        try:
            self.assertEqual(hc.main(["ping"]), 1)
        finally:
            hc.DEFAULT_BASE_URL = original


if __name__ == "__main__":
    unittest.main()
