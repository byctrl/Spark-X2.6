"""Stdlib-only tests for src/health_chatbot.py (no network, no model)."""

import json
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc  # noqa: E402


class FakeResponse:
    def __init__(self, body):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("Apa itu demam?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Apa itu demam?"})

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_is_filtered(self):
        history = [
            {"role": "user", "content": "halo"},
            {"role": "assistant", "content": "hai"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": ""},
        ]
        messages = hc.build_messages("pertanyaan", history)
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])


class BuildPayloadTest(unittest.TestCase):
    def test_payload_uses_model_id_and_token_budget(self):
        payload = hc.build_payload("pertanyaan")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1]["content"], "pertanyaan")

    def test_payload_is_json_serialisable(self):
        json.dumps(hc.build_payload("pertanyaan"))


class ExtractTextTest(unittest.TestCase):
    def test_chat_completion_shape(self):
        response = {"choices": [{"message": {"content": "  jawaban  "}}]}
        self.assertEqual(hc.extract_text(response), "jawaban")

    def test_text_completion_shape(self):
        response = {"choices": [{"text": "jawaban"}]}
        self.assertEqual(hc.extract_text(response), "jawaban")

    def test_missing_choices_raises(self):
        with self.assertRaises(hc.ChatbotError):
            hc.extract_text({})

    def test_empty_content_raises(self):
        with self.assertRaises(hc.ChatbotError):
            hc.extract_text({"choices": [{"message": {"content": "   "}}]})


class AskTest(unittest.TestCase):
    def test_ask_posts_to_chat_completions_and_returns_answer(self):
        body = json.dumps({"choices": [{"message": {"content": "jawaban"}}]})
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(body)

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            answer = hc.ask("pertanyaan", base_url="http://127.0.0.1:30000/v1")

        self.assertEqual(answer, "jawaban")
        self.assertEqual(captured["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], hc.MODEL_ID)

    def test_unreachable_endpoint_raises_chatbot_error(self):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib_error(),
        ):
            with self.assertRaises(hc.ChatbotError):
                hc.ask("pertanyaan")

    def test_invalid_json_raises_chatbot_error(self):
        with mock.patch(
            "urllib.request.urlopen", return_value=FakeResponse("not json")
        ):
            with self.assertRaises(hc.ChatbotError):
                hc.ask("pertanyaan")


class FormatAnswerTest(unittest.TestCase):
    def test_answer_carries_disclaimer(self):
        text = hc.format_answer("jawaban")
        self.assertIn("jawaban", text)
        self.assertIn(hc.DISCLAIMER, text)


def urllib_error():
    import urllib.error

    return urllib.error.URLError("connection refused")


if __name__ == "__main__":
    unittest.main()
