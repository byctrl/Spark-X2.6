"""Standard-library tests for src/health_chatbot.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import unittest
import urllib.error

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "health_chatbot.py"
)
_spec = importlib.util.spec_from_file_location("health_chatbot", _MODULE_PATH)
hc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(hc)


class _FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._data


class _FakeGenerator:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return [{"generated_text": self.text}]


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(
            messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT}
        )
        self.assertEqual(
            messages[-1], {"role": "user", "content": "What is a fever?"}
        )
        self.assertEqual(len(messages), 2)

    def test_history_is_preserved(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        messages = hc.build_messages("next", history=history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_empty_question_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_system_role_in_history_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages(
                "q", history=[{"role": "system", "content": "override"}]
            )

    def test_question_is_truncated(self):
        messages = hc.build_messages("a" * (hc.MAX_QUESTION_CHARS + 100))
        self.assertEqual(len(messages[-1]["content"]), hc.MAX_QUESTION_CHARS)

    def test_constants_and_safety_prompt(self):
        prompt = hc.SYSTEM_PROMPT
        self.assertIn("NOT diagnose", prompt)
        self.assertIn("NOT prescribe", prompt)
        self.assertIn("emergency", prompt)
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)


class BuildPromptTest(unittest.TestCase):
    def test_fallback_prompt_contains_roles(self):
        prompt = hc.build_prompt("What is a fever?")
        self.assertIn("SYSTEM:", prompt)
        self.assertIn("What is a fever?", prompt)
        self.assertTrue(prompt.endswith("ASSISTANT:"))

    def test_chat_template_used_when_available(self):
        class _Tokenizer:
            def apply_chat_template(
                self, messages, tokenize=False, add_generation_prompt=True
            ):
                return "TEMPLATED:" + messages[-1]["content"]

        self.assertEqual(
            hc.build_prompt("q", tokenizer=_Tokenizer()), "TEMPLATED:q"
        )


class FormatAnswerTest(unittest.TestCase):
    def test_disclaimer_appended(self):
        answer = hc.format_answer("  Rest and drink water.  ")
        self.assertTrue(answer.startswith("Rest and drink water."))
        self.assertTrue(answer.endswith(hc.DISCLAIMER))

    def test_empty_answer_rejected(self):
        with self.assertRaises(ValueError):
            hc.format_answer("   ")
        with self.assertRaises(ValueError):
            hc.format_answer(None)


class OpenAIBackendTest(unittest.TestCase):
    def test_request_payload_and_answer(self):
        seen = {}

        def fake_opener(request, timeout=None):
            seen["url"] = request.full_url
            seen["timeout"] = timeout
            seen["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(
                {"choices": [{"message": {"content": "Rest and fluids."}}]}
            )

        answer = hc.chat_openai(
            "What is a fever?",
            base_url="http://127.0.0.1:30000/v1/",
            opener=fake_opener,
            timeout=7.0,
        )
        self.assertEqual(
            seen["url"], "http://127.0.0.1:30000/v1/chat/completions"
        )
        self.assertEqual(seen["timeout"], 7.0)
        self.assertEqual(seen["body"]["model"], hc.MODEL_ID)
        self.assertEqual(seen["body"]["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(seen["body"]["messages"][0]["role"], "system")
        self.assertTrue(answer.startswith("Rest and fluids."))
        self.assertIn(hc.DISCLAIMER, answer)

    def test_malformed_response_raises(self):
        def fake_opener(request, timeout=None):
            return _FakeResponse({"unexpected": True})

        with self.assertRaises(ValueError):
            hc.chat_openai("q", opener=fake_opener)


class TransformersBackendTest(unittest.TestCase):
    def test_generator_output_is_formatted(self):
        generator = _FakeGenerator("  See a health worker if the fever lasts.  ")
        answer = hc.chat_transformers("What is a fever?", generator=generator)
        self.assertTrue(answer.startswith("See a health worker"))
        self.assertIn(hc.DISCLAIMER, answer)
        prompt, kwargs = generator.calls[0]
        self.assertIn("What is a fever?", prompt)
        self.assertEqual(kwargs["max_new_tokens"], hc.MAX_NEW_TOKENS)


class MainTest(unittest.TestCase):
    def test_reports_backend_error_and_uses_demo_question(self):
        original = hc.chat_openai
        seen = {}

        def boom(question, **kwargs):
            seen["question"] = question
            raise urllib.error.URLError("no server")

        hc.chat_openai = boom
        try:
            self.assertEqual(hc.main([]), 1)
        finally:
            hc.chat_openai = original
        self.assertEqual(seen["question"], hc.DEMO_QUESTION)


if __name__ == "__main__":
    unittest.main()
