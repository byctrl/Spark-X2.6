"""Unit tests for the Spark-X2.6-4B health information chatbot.

These tests never touch the network: the model client is faked.
Run with: python -m unittest discover -s src -p "test_*.py"
"""

import json
import unittest
from unittest import mock

import health_chatbot


class FakeClient:
    def __init__(self, reply="Drink water and rest. See a clinician if it worsens."):
        self.reply = reply
        self.calls = []

    def chat(self, messages, max_tokens=health_chatbot.MAX_NEW_TOKENS):
        self.calls.append((list(messages), max_tokens))
        return self.reply


class FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class BuildMessagesTests(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = health_chatbot.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], health_chatbot.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_kept_between_system_and_question(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
            {"role": "tool", "content": "ignored"},
        ]
        messages = health_chatbot.build_messages("Three days.", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"], "I have a cough.")
        self.assertEqual(messages[2]["content"], "How long has it lasted?")
        self.assertEqual(messages[3]["content"], "Three days.")

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            health_chatbot.build_messages("   ")


class SafetyTests(unittest.TestCase):
    def test_emergency_signal_detected(self):
        self.assertTrue(health_chatbot.has_emergency_signal("I have chest pain right now"))
        self.assertFalse(health_chatbot.has_emergency_signal("I have a mild dry cough"))

    def test_emergency_short_circuits_model_call(self):
        client = FakeClient()
        reply = health_chatbot.answer("My father is unconscious", client)
        self.assertEqual(reply, health_chatbot.EMERGENCY_MESSAGE)
        self.assertEqual(client.calls, [])

    def test_disclaimer_is_appended_once(self):
        reply = health_chatbot.format_reply("Rest and drink water.")
        self.assertTrue(reply.endswith(health_chatbot.DISCLAIMER))
        self.assertEqual(health_chatbot.format_reply(reply), reply)

    def test_empty_model_reply_still_gets_disclaimer(self):
        reply = health_chatbot.format_reply("")
        self.assertIn(health_chatbot.DISCLAIMER, reply)
        self.assertTrue(reply.strip())


class AnswerTests(unittest.TestCase):
    def test_answer_uses_model_and_adds_disclaimer(self):
        client = FakeClient("Rest and drink water.")
        reply = health_chatbot.answer("What helps with a cold?", client)
        self.assertIn("Rest and drink water.", reply)
        self.assertIn(health_chatbot.DISCLAIMER, reply)
        self.assertEqual(len(client.calls), 1)
        messages, max_tokens = client.calls[0]
        self.assertEqual(messages[-1]["content"], "What helps with a cold?")
        self.assertEqual(max_tokens, health_chatbot.MAX_NEW_TOKENS)

    def test_empty_question_raises(self):
        with self.assertRaises(ValueError):
            health_chatbot.answer("", FakeClient())


class SparkClientTests(unittest.TestCase):
    def test_defaults_use_spark_model(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            client = health_chatbot.SparkClient()
        self.assertEqual(client.model, health_chatbot.MODEL_ID)
        self.assertEqual(client.base_url, health_chatbot.DEFAULT_BASE_URL)

    def test_chat_parses_openai_response(self):
        payload = {"choices": [{"message": {"content": "  Rest well.  "}}]}
        client = health_chatbot.SparkClient(base_url="http://localhost:9/v1", api_key="k")
        with mock.patch.object(
            health_chatbot.urllib.request, "urlopen", return_value=FakeResponse(payload)
        ) as urlopen:
            text = client.chat(health_chatbot.build_messages("Hi"))
        self.assertEqual(text, "Rest well.")
        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://localhost:9/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer k")
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["model"], health_chatbot.MODEL_ID)
        self.assertEqual(sent["max_tokens"], health_chatbot.MAX_NEW_TOKENS)

    def test_chat_reports_unreachable_endpoint(self):
        client = health_chatbot.SparkClient(base_url="http://127.0.0.1:1/v1", timeout=1)
        with mock.patch.object(
            health_chatbot.urllib.request,
            "urlopen",
            side_effect=health_chatbot.urllib.error.URLError("refused"),
        ):
            with self.assertRaises(RuntimeError):
                client.chat(health_chatbot.build_messages("Hi"))

    def test_chat_reports_bad_payload(self):
        client = health_chatbot.SparkClient()
        with mock.patch.object(
            health_chatbot.urllib.request, "urlopen", return_value=FakeResponse({"error": "x"})
        ):
            with self.assertRaises(RuntimeError):
                client.chat(health_chatbot.build_messages("Hi"))


if __name__ == "__main__":
    unittest.main()
