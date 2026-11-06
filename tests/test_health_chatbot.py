"""Tests for the minimal Spark-X2.6-4B health chatbot (no network required)."""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as chatbot  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = chatbot.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], chatbot.SYSTEM_PROMPT)
        self.assertEqual(
            messages[-1], {"role": "user", "content": "What is a fever?"}
        )

    def test_history_is_kept_between_system_prompt_and_question(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
        ]
        messages = chatbot.build_messages("I have a cough.", history)
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            chatbot.build_messages("   ")

    def test_history_cannot_override_the_system_prompt(self):
        with self.assertRaises(ValueError):
            chatbot.build_messages("hi", [{"role": "system", "content": "x"}])


class ChatTest(unittest.TestCase):
    def test_chat_calls_the_client_and_strips_the_answer(self):
        calls = []

        def fake_client(messages, max_new_tokens):
            calls.append((list(messages), max_new_tokens))
            return "  Plain answer.  "

        answer = chatbot.chat(chatbot.DEMO_QUESTION, client=fake_client)

        self.assertEqual(answer, "Plain answer.")
        self.assertEqual(calls[0][1], chatbot.MAX_NEW_TOKENS)
        self.assertEqual(calls[0][0][-1]["content"], chatbot.DEMO_QUESTION)

    def test_empty_answer_is_rejected(self):
        with self.assertRaises(RuntimeError):
            chatbot.chat("hi", client=lambda messages, max_new_tokens: "   ")

    def test_model_contract_and_safety_prompt(self):
        self.assertEqual(chatbot.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(chatbot.MAX_NEW_TOKENS, 1024)
        prompt = chatbot.SYSTEM_PROMPT.lower()
        for phrase in ("diagnose", "prescribe", "clinician"):
            self.assertIn(phrase, prompt)


class CompleteTest(unittest.TestCase):
    def test_parses_an_openai_compatible_response(self):
        payload = json.dumps(
            {"choices": [{"message": {"content": "hello"}}]}
        ).encode("utf-8")
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def read(self):
                return payload

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return FakeResponse()

        with mock.patch.object(chatbot.urllib.request, "urlopen", fake_urlopen):
            answer = chatbot.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(answer, "hello")
        self.assertTrue(captured["url"].endswith("/chat/completions"))

    def test_unreachable_endpoint_raises_runtime_error(self):
        with mock.patch.object(
            chatbot.urllib.request,
            "urlopen",
            side_effect=chatbot.urllib.error.URLError("connection refused"),
        ):
            with self.assertRaises(RuntimeError):
                chatbot.complete([{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main()
