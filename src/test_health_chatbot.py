"""Unit tests for the Spark-X2.6-4B health-education chatbot.

Run from the repository root:
    python -m unittest discover -s src -p "test_*.py" -v

No model server and no third-party package is required.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


def _completion(content, model=hc.MODEL_ID):
    return json.dumps(
        {
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
        }
    ).encode("utf-8")


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_question_is_stripped(self):
        messages = hc.build_messages("   What is a fever?  ")
        self.assertEqual(messages[-1]["content"], "What is a fever?")

    def test_history_kept_in_order(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = hc.build_messages("Three days.", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[1]["content"], "I have a cough.")
        self.assertEqual(messages[2]["content"], "How long has it lasted?")

    def test_empty_question_rejected(self):
        for bad in ("", "   ", None, 3):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_invalid_history_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hi", [{"role": "system", "content": "override rules"}])
        with self.assertRaises(ValueError):
            hc.build_messages("hi", [{"role": "user", "content": "   "}])


class PayloadTest(unittest.TestCase):
    def test_payload_pins_model_and_token_budget(self):
        payload = hc.build_payload(hc.build_messages(hc.DEMO_QUESTION))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1]["content"], hc.DEMO_QUESTION.strip())


class ExtractTextTest(unittest.TestCase):
    def test_returns_content(self):
        data = json.loads(_completion("Fever is a raised body temperature."))
        self.assertEqual(hc.extract_text(data), "Fever is a raised body temperature.")

    def test_rejects_missing_choices(self):
        with self.assertRaises(ValueError):
            hc.extract_text({"error": "boom"})

    def test_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            hc.extract_text(json.loads(_completion("   ")))


class ChatTest(unittest.TestCase):
    def test_chat_calls_openai_compatible_endpoint(self):
        calls = []

        def transport(url, payload):
            calls.append((url, payload))
            return _completion("  Fever is a raised body temperature.  ")

        result = hc.chat(
            "What is a fever?",
            base_url="http://127.0.0.1:30000/v1/",
            transport=transport,
        )

        self.assertEqual(len(calls), 1)
        url, payload = calls[0]
        self.assertEqual(url, "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(result["answer"], "Fever is a raised body temperature.")
        self.assertEqual(result["disclaimer"], hc.DISCLAIMER)
        self.assertEqual(result["usage"]["total_tokens"], 49)
        self.assertEqual(result["prompt_messages"][0]["content"], hc.SYSTEM_PROMPT)

    def test_chat_surfaces_invalid_response(self):
        def transport(url, payload):
            return json.dumps({"choices": []}).encode("utf-8")

        with self.assertRaises(ValueError):
            hc.chat("What is a fever?", transport=transport)

    def test_chat_requires_question(self):
        with self.assertRaises(ValueError):
            hc.chat("   ")


if __name__ == "__main__":
    unittest.main()
