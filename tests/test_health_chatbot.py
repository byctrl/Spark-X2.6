"""Offline unit tests for the Spark-X2.6-4B health-information chatbot.

These tests never touch the network: they only check prompt construction,
safety framing and response normalisation.
"""

import unittest

from src.health_chatbot import (
    DEFAULT_BASE_URL,
    DEMO_QUESTION,
    DISCLAIMER,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    SparkClient,
    build_messages,
    build_payload,
    build_prompt,
    clean_reply,
    with_disclaimer,
)


class ModelConfigTest(unittest.TestCase):
    def test_model_id_is_spark_x26_4b(self):
        self.assertEqual(MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_token_budget(self):
        self.assertEqual(MAX_NEW_TOKENS, 1024)

    def test_system_prompt_forbids_diagnosis(self):
        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", lowered)
        self.assertIn("clinician", lowered)

    def test_demo_question_is_present(self):
        self.assertTrue(DEMO_QUESTION.strip())


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = build_messages("How do I read this lab result?")
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})

    def test_user_turn_is_last_and_trimmed(self):
        messages = build_messages("  What is a fever?  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_preserved_in_order(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "should be ignored"},
            {"role": "assistant", "content": "   "},
            "not a dict",
        ]
        messages = build_messages("next", history=history)
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                build_messages(bad)


class PayloadTest(unittest.TestCase):
    def test_payload_matches_openai_chat_shape(self):
        payload = build_payload("What is a fever?")
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertEqual(payload["max_tokens"], MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1]["role"], "user")

    def test_payload_is_json_serialisable(self):
        import json

        encoded = json.dumps(build_payload("What is a fever?"))
        self.assertIn("Spark-X2.6-4B", encoded)


class PromptTest(unittest.TestCase):
    def test_prompt_ends_with_assistant_tag(self):
        prompt = build_prompt(build_messages("What is a fever?"))
        self.assertTrue(prompt.endswith("<|assistant|>\n"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)


class ReplyTest(unittest.TestCase):
    def test_clean_reply_strips_assistant_prefix(self):
        self.assertEqual(clean_reply("  assistant: Drink water.  "), "Drink water.")

    def test_clean_reply_truncates(self):
        cleaned = clean_reply("x" * 50, max_chars=10)
        self.assertTrue(cleaned.startswith("x" * 10))
        self.assertTrue(cleaned.endswith("..."))

    def test_clean_reply_rejects_non_string(self):
        with self.assertRaises(ValueError):
            clean_reply(None)

    def test_disclaimer_is_appended_once(self):
        once = with_disclaimer("Rest and monitor your temperature.")
        self.assertIn(DISCLAIMER, once)
        twice = with_disclaimer(once)
        self.assertEqual(twice, once)

    def test_empty_reply_becomes_disclaimer(self):
        self.assertEqual(with_disclaimer("   "), DISCLAIMER)


class ClientConfigTest(unittest.TestCase):
    def test_default_endpoint(self):
        client = SparkClient(base_url=None, model=MODEL_ID)
        self.assertTrue(client.endpoint().endswith("/chat/completions"))
        self.assertIn("127.0.0.1:30000", DEFAULT_BASE_URL)

    def test_trailing_slash_is_normalised(self):
        client = SparkClient(base_url="http://127.0.0.1:30000/v1/")
        self.assertEqual(client.endpoint(), "http://127.0.0.1:30000/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
