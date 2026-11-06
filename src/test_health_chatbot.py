"""Tests for the Spark-X2.6-4B health-education chatbot (no network)."""

import json
import unittest

from health_chatbot import (
    DEMO_QUESTION,
    EMERGENCY_NOTICE,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    build_messages,
    generate_reply,
    needs_emergency_notice,
)


def fake_transport(reply_text, captured=None):
    def transport(url, payload, timeout):
        if captured is not None:
            captured.append((url, payload, timeout))
        body = {"choices": [{"message": {"content": reply_text}}]}
        return json.dumps(body).encode("utf-8")

    return transport


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = build_messages("  what is a fever?  ")
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "what is a fever?"})

    def test_system_prompt_forbids_diagnosis_and_prescription(self):
        self.assertIn("Do not diagnose", SYSTEM_PROMPT)
        self.assertIn("do not prescribe", SYSTEM_PROMPT)
        self.assertIn("qualified clinician", SYSTEM_PROMPT)

    def test_history_is_preserved(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = build_messages("Four days.", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("   ")

    def test_bad_history_role_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("hi", [{"role": "system", "content": "x"}])


class GenerateReplyTest(unittest.TestCase):
    def test_request_uses_model_and_token_limit(self):
        captured = []
        reply = generate_reply(
            DEMO_QUESTION,
            transport=fake_transport("Ask your doctor about it.", captured),
        )
        self.assertEqual(reply, "Ask your doctor about it.")
        url, payload, _timeout = captured[0]
        self.assertTrue(url.endswith("/chat/completions"))
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertEqual(payload["max_tokens"], MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1]["content"], DEMO_QUESTION)

    def test_custom_base_url_is_normalised(self):
        captured = []
        generate_reply(
            "hello",
            base_url="http://127.0.0.1:8080/v1/",
            transport=fake_transport("ok", captured),
        )
        self.assertEqual(captured[0][0], "http://127.0.0.1:8080/v1/chat/completions")

    def test_emergency_notice_is_prepended(self):
        reply = generate_reply(
            "I have chest pain right now",
            transport=fake_transport("Seek care soon."),
        )
        self.assertTrue(reply.startswith(EMERGENCY_NOTICE))
        self.assertIn("Seek care soon.", reply)

    def test_emergency_notice_not_duplicated(self):
        text = EMERGENCY_NOTICE + "\n\nSeek care soon."
        reply = generate_reply("chest pain", transport=fake_transport(text))
        self.assertEqual(reply.count(EMERGENCY_NOTICE), 1)

    def test_empty_answer_raises(self):
        with self.assertRaises(RuntimeError):
            generate_reply("hello", transport=fake_transport("   "))

    def test_missing_choices_raises(self):
        def transport(url, payload, timeout):
            return b"{}"

        with self.assertRaises(RuntimeError):
            generate_reply("hello", transport=transport)

    def test_non_json_raises(self):
        def transport(url, payload, timeout):
            return b"not json"

        with self.assertRaises(RuntimeError):
            generate_reply("hello", transport=transport)


class EmergencyScreenTest(unittest.TestCase):
    def test_keywords_detected_case_insensitively(self):
        self.assertTrue(needs_emergency_notice("Severe Bleeding after a fall"))
        self.assertTrue(needs_emergency_notice("I think it is a stroke"))

    def test_routine_question_not_flagged(self):
        self.assertFalse(needs_emergency_notice(DEMO_QUESTION))
        self.assertFalse(needs_emergency_notice(""))


if __name__ == "__main__":
    unittest.main()
