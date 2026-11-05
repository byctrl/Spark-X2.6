"""Offline tests for src/health_chatbot.py (no network, no model)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import health_chatbot as hc  # noqa: E402


class RecordingPoster:
    """Stand-in for post_json that records the request instead of sending it."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, payload, timeout=None):
        self.calls.append({"url": url, "payload": payload})
        return self.response


def reply(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class BuildMessagesTests(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_system_prompt_forbids_diagnosis_and_prescribing(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("diagnose", prompt)
        self.assertIn("prescribe", prompt)
        self.assertIn("clinician", prompt)

    def test_history_is_kept_and_junk_dropped(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
            {"role": "system", "content": "ignored"},
            "not-a-dict",
            {"role": "assistant", "content": "   "},
        ]
        messages = hc.build_messages("Two weeks.", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"], "I have a cough.")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], "Two weeks.")


class PayloadTests(unittest.TestCase):
    def test_payload_uses_spark_model_and_token_budget(self):
        payload = hc.build_payload(hc.build_messages("hello"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])


class AskTests(unittest.TestCase):
    def test_ask_posts_to_chat_completions_and_returns_answer(self):
        poster = RecordingPoster(reply("  See a clinician if the fever lasts.  "))
        answer = hc.ask("I have a cough.", base_url="http://127.0.0.1:30000/v1/", _post=poster)
        self.assertEqual(answer, "See a clinician if the fever lasts.")
        self.assertEqual(len(poster.calls), 1)
        self.assertEqual(
            poster.calls[0]["url"], "http://127.0.0.1:30000/v1/chat/completions"
        )
        self.assertEqual(poster.calls[0]["payload"]["model"], hc.MODEL_ID)

    def test_ask_rejects_blank_question(self):
        poster = RecordingPoster(reply("unused"))
        for bad in ["", "   ", None]:
            with self.subTest(question=bad):
                with self.assertRaises(ValueError):
                    hc.ask(bad, _post=poster)
        self.assertEqual(poster.calls, [])

    def test_ask_rejects_empty_model_output(self):
        with self.assertRaises(ValueError):
            hc.ask("hello", _post=RecordingPoster(reply("   ")))

    def test_ask_rejects_malformed_response(self):
        with self.assertRaises(ValueError):
            hc.ask("hello", _post=RecordingPoster({"choices": []}))


if __name__ == "__main__":
    unittest.main()
