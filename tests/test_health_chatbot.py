"""Small stdlib-only tests for the Spark-X2.6-4B health chatbot demo."""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    ),
)

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_starts_with_system_prompt(self):
        msgs = hc.build_messages("How do I treat a fever?")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("Never diagnose", msgs[0]["content"])
        self.assertIn("not medical advice", msgs[0]["content"])

    def test_last_message_is_the_trimmed_question(self):
        msgs = hc.build_messages("  cough for three days  ")
        self.assertEqual(msgs[-1], {"role": "user", "content": "cough for three days"})

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_kept_and_invalid_turns_dropped(self):
        msgs = hc.build_messages(
            "next",
            history=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "ignored"},
                {"role": "user", "content": "   "},
                "not-a-dict",
            ],
        )
        self.assertEqual(
            [m["role"] for m in msgs], ["system", "user", "assistant", "user"]
        )


class PayloadTest(unittest.TestCase):
    def test_payload_uses_spark_model_and_no_stream(self):
        payload = hc.build_payload(hc.build_messages("q"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])


class ParseCompletionTest(unittest.TestCase):
    def test_extracts_and_strips_content(self):
        data = {"choices": [{"message": {"role": "assistant", "content": " Drink water. "}}]}
        self.assertEqual(hc.parse_completion(data), "Drink water.")

    def test_missing_choices_raises(self):
        with self.assertRaises(ValueError):
            hc.parse_completion({})

    def test_empty_content_raises(self):
        with self.assertRaises(ValueError):
            hc.parse_completion({"choices": [{"message": {"content": "  "}}]})


class AskTest(unittest.TestCase):
    def test_ask_posts_to_chat_completions_and_returns_answer(self):
        captured = {}

        def fake_transport(url, payload, api_key, timeout):
            captured["url"] = url
            captured["payload"] = payload
            captured["api_key"] = api_key
            return {
                "choices": [
                    {"message": {"content": "Ask a clinician about the fever."}}
                ]
            }

        answer = hc.ask(
            "fever after travel",
            base_url="http://localhost:9/v1/",
            api_key="k",
            transport=fake_transport,
        )

        self.assertEqual(answer, "Ask a clinician about the fever.")
        self.assertEqual(captured["url"], "http://localhost:9/v1/chat/completions")
        self.assertEqual(captured["api_key"], "k")
        self.assertEqual(
            captured["payload"]["messages"][-1],
            {"role": "user", "content": "fever after travel"},
        )


if __name__ == "__main__":
    unittest.main()
