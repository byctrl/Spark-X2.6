"""Tests for the Spark-X2.6-4B health chatbot demo (no network needed)."""

import json
import unittest

import health_chatbot as hc


class _FakeResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class PromptTests(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  what is a fever?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "what is a fever?"})

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_rejects_bad_history(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hi", history=[{"role": "system", "content": "x"}])

    def test_payload_uses_model_id_and_token_budget(self):
        payload = hc.build_payload("hello")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])

    def test_system_prompt_keeps_safety_boundary(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", lowered)
        self.assertIn("clinician", lowered)


class ReplyTests(unittest.TestCase):
    def test_extract_reply_reads_first_choice(self):
        data = {"choices": [{"message": {"content": "  Wear a mask. Next step: see a clinician.  "}}]}
        self.assertEqual(
            hc.extract_reply(data), "Wear a mask. Next step: see a clinician."
        )

    def test_extract_reply_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            hc.extract_reply({"choices": []})

    def test_extract_reply_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            hc.extract_reply({"choices": [{"message": {"content": "   "}}]})


class AskTests(unittest.TestCase):
    def test_ask_posts_chat_completion_and_returns_reply(self):
        seen = {}

        def opener(request, timeout=None):
            seen["url"] = request.full_url
            seen["body"] = json.loads(request.data.decode("utf-8"))
            seen["timeout"] = timeout
            return _FakeResponse(
                {"choices": [{"message": {"content": "ok. Next step: see a clinician."}}]}
            )

        reply = hc.ask("cough", base_url="http://127.0.0.1:30000/v1", opener=opener)
        self.assertEqual(reply, "ok. Next step: see a clinician.")
        self.assertEqual(seen["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(seen["body"]["model"], hc.MODEL_ID)
        self.assertEqual(seen["body"]["messages"][-1]["content"], "cough")


if __name__ == "__main__":
    unittest.main()
