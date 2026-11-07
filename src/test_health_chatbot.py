"""Small offline tests for the Spark-X2.6-4B health chatbot demo."""

import unittest

import health_chatbot


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_forbids_diagnosis(self):
        prompt = health_chatbot.SYSTEM_PROMPT.lower()
        self.assertIn("diagnose", prompt)
        self.assertIn("prescribe", prompt)

    def test_question_is_trimmed_and_last(self):
        messages = health_chatbot.build_messages("  I have a cough  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "I have a cough"})

    def test_history_turns_are_kept(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignored"},
        ]
        messages = health_chatbot.build_messages("next", history)
        roles = [message["role"] for message in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            health_chatbot.build_messages("   ")


class PayloadTest(unittest.TestCase):
    def test_payload_uses_model_and_token_cap(self):
        messages = health_chatbot.build_messages("cough")
        payload = health_chatbot.build_payload(messages)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], health_chatbot.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])


class ExtractReplyTest(unittest.TestCase):
    def test_reads_message_content(self):
        data = {"choices": [{"message": {"role": "assistant", "content": " ok "}}]}
        self.assertEqual(health_chatbot.extract_reply(data), "ok")

    def test_rejects_empty_choices(self):
        with self.assertRaises(ValueError):
            health_chatbot.extract_reply({"choices": []})


class SafetyNoteTest(unittest.TestCase):
    def test_note_added_when_no_referral(self):
        reply = health_chatbot.with_safety_note("Rest and drink water.")
        self.assertIn("clinician", reply.lower())

    def test_note_skipped_when_referral_present(self):
        text = "Please call your doctor or emergency services."
        self.assertEqual(health_chatbot.with_safety_note(text), text)


class ChatTest(unittest.TestCase):
    def test_chat_calls_openai_endpoint(self):
        captured = {}
        original = health_chatbot._post_json

        def fake_post(url, payload, timeout=health_chatbot.REQUEST_TIMEOUT):
            captured["url"] = url
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "Rest and drink water."}}]}

        health_chatbot._post_json = fake_post
        try:
            reply = health_chatbot.chat(
                "cough", base_url="http://127.0.0.1:30000/v1/"
            )
        finally:
            health_chatbot._post_json = original

        self.assertEqual(captured["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], health_chatbot.MODEL_ID)
        self.assertIn("clinician", reply.lower())


if __name__ == "__main__":
    unittest.main()
