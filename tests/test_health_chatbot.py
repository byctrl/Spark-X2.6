"""Offline tests for the Spark-X2.6-4B health chatbot demo (no network needed)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import health_chatbot as hc


class ValidateQuestionTest(unittest.TestCase):
    def test_returns_stripped_question(self):
        self.assertEqual(hc.validate_question("  apa itu demam?  "), "apa itu demam?")

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            hc.validate_question("   ")

    def test_rejects_non_string(self):
        with self.assertRaises(ValueError):
            hc.validate_question(None)

    def test_rejects_over_long_question(self):
        with self.assertRaises(ValueError):
            hc.validate_question("a" * (hc.MAX_QUESTION_CHARS + 1))


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  apa itu demam?  ")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "apa itu demam?"})

    def test_history_is_filtered_and_kept_in_order(self):
        history = [
            {"role": "user", "content": "  halo  "},
            {"role": "assistant", "content": "halo juga"},
            {"role": "tool", "content": "ignored"},
            {"role": "user", "content": "   "},
            "not-a-dict",
        ]
        messages = hc.build_messages("pertanyaan baru", history=history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "halo")
        self.assertEqual(messages[-1]["content"], "pertanyaan baru")

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            hc.build_messages("")


class ExtractMessageTextTest(unittest.TestCase):
    def test_reads_plain_content(self):
        response = {"choices": [{"message": {"role": "assistant", "content": " hai "}}]}
        self.assertEqual(hc.extract_message_text(response), "hai")

    def test_joins_content_parts(self):
        response = {
            "choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]
        }
        self.assertEqual(hc.extract_message_text(response), "ab")

    def test_rejects_missing_choices(self):
        with self.assertRaises(hc.SparkError):
            hc.extract_message_text({})

    def test_rejects_empty_content(self):
        with self.assertRaises(hc.SparkError):
            hc.extract_message_text({"choices": [{"message": {"content": "   "}}]})


class SparkChatClientTest(unittest.TestCase):
    def test_payload_endpoint_and_reply(self):
        captured = {}

        def transport(url, payload, timeout):
            captured["url"] = url
            captured["payload"] = payload
            captured["timeout"] = timeout
            return {"choices": [{"message": {"content": " jawaban "}}]}

        client = hc.SparkChatClient(
            base_url="http://127.0.0.1:9/v1/", transport=transport, timeout=5
        )
        reply = client.chat(hc.build_messages("halo"))

        self.assertEqual(captured["url"], "http://127.0.0.1:9/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(captured["payload"]["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(captured["payload"]["stream"])
        self.assertEqual(captured["payload"]["messages"][0]["role"], "system")
        self.assertEqual(captured["timeout"], 5)
        self.assertEqual(reply, "jawaban")

    def test_propagates_transport_errors(self):
        def transport(url, payload, timeout):
            raise hc.SparkError("boom")

        client = hc.SparkChatClient(base_url="http://127.0.0.1:9/v1", transport=transport)
        with self.assertRaises(hc.SparkError):
            client.chat(hc.build_messages("halo"))


class StubClient:
    def __init__(self, reply="Ringkasan."):
        self.reply = reply
        self.messages = None

    def chat(self, messages):
        self.messages = messages
        return self.reply


class AnswerQuestionTest(unittest.TestCase):
    def test_appends_disclaimer_and_sends_question(self):
        stub = StubClient()
        result = hc.answer_question("apa itu demam?", client=stub)
        self.assertTrue(result.startswith("Ringkasan."))
        self.assertIn(hc.DISCLAIMER, result)
        self.assertEqual(stub.messages[-1]["content"], "apa itu demam?")

    def test_rejects_empty_question_before_calling_model(self):
        stub = StubClient()
        with self.assertRaises(ValueError):
            hc.answer_question("   ", client=stub)
        self.assertIsNone(stub.messages)


if __name__ == "__main__":
    unittest.main()
