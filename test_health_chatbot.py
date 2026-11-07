"""Offline tests for the Spark-X2.6-4B health-information chatbot.

These tests never touch the network and never need model weights.
Run from the repository root::

    python -m unittest test_health_chatbot -v
"""

import json
import unittest

import health_chatbot as chatbot


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = chatbot.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], chatbot.SYSTEM_PROMPT)
        self.assertIn("Never diagnose", chatbot.SYSTEM_PROMPT)

    def test_user_question_is_last(self):
        messages = chatbot.build_messages("  When should I see a doctor?  ")
        self.assertEqual(messages[-1], {
            "role": "user",
            "content": "When should I see a doctor?",
        })

    def test_history_is_kept_and_filtered(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
            {"role": "system", "content": "ignored"},
            {"role": "assistant", "content": ""},
        ]
        messages = chatbot.build_messages("And now?", history)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            chatbot.build_messages("   ")


class PayloadTest(unittest.TestCase):
    def test_payload_uses_the_model_and_token_budget(self):
        messages = chatbot.build_messages("How much water should I drink?")
        payload = chatbot.build_payload(messages)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], chatbot.MAX_NEW_TOKENS)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"], messages)

    def test_request_targets_chat_completions(self):
        payload = chatbot.build_payload(chatbot.build_messages("Hi"))
        request = chatbot.build_request("http://127.0.0.1:30000/v1/", payload)
        self.assertEqual(request.full_url,
                         "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data.decode("utf-8")), payload)


class ExtractReplyTest(unittest.TestCase):
    def test_reads_assistant_content(self):
        response = {"choices": [{"message": {"content": " Rest well. "}}]}
        self.assertEqual(chatbot.extract_reply(response), "Rest well.")

    def test_rejects_unexpected_shape(self):
        with self.assertRaises(ValueError):
            chatbot.extract_reply({"error": "boom"})

    def test_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            chatbot.extract_reply({"choices": [{"message": {"content": "  "}}]})


class SafetyNoteTest(unittest.TestCase):
    def test_note_added_when_no_clinician_is_mentioned(self):
        answer = chatbot.with_safety_note("Drink fluids and rest.")
        self.assertTrue(answer.startswith("Drink fluids and rest."))
        self.assertIn(chatbot.SAFETY_NOTE, answer)

    def test_note_skipped_when_clinician_is_mentioned(self):
        answer = "Please see a doctor if the fever lasts more than three days."
        self.assertEqual(chatbot.with_safety_note(answer), answer)

    def test_note_not_duplicated(self):
        once = chatbot.with_safety_note("Drink fluids and rest.")
        self.assertEqual(chatbot.with_safety_note(once), once)


class ClientTest(unittest.TestCase):
    def test_default_base_url_is_local(self):
        client = chatbot.SparkClient()
        self.assertTrue(client.base_url.startswith("http://"))
        self.assertFalse(client.base_url.endswith("/"))


if __name__ == "__main__":
    unittest.main()
