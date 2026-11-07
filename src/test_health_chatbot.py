"""Offline unit tests for src/health_chatbot.py (no model, no network)."""

import os
import sys
import unittest
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from health_chatbot import (  # noqa: E402
    DEMO_QUESTION,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    build_messages,
    build_payload,
    chat,
    needs_emergency_notice,
    parse_response,
)


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = build_messages(DEMO_QUESTION)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": DEMO_QUESTION})

    def test_prompt_forbids_diagnosis(self):
        self.assertIn("Never give a diagnosis", SYSTEM_PROMPT)

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("   ")


class BuildPayloadTest(unittest.TestCase):
    def test_payload_targets_spark_model(self):
        payload = build_payload(build_messages(DEMO_QUESTION))
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1]["content"], DEMO_QUESTION)


class ParseResponseTest(unittest.TestCase):
    def test_reads_message_content(self):
        data = {"choices": [{"message": {"content": "  Halo.  "}, "finish_reason": "stop"}]}
        self.assertEqual(parse_response(data), "Halo.")

    def test_falls_back_to_reasoning_content(self):
        data = {"choices": [{"message": {"content": None, "reasoning_content": "Penjelasan."}}]}
        self.assertEqual(parse_response(data), "Penjelasan.")

    def test_empty_content_raises(self):
        data = {"choices": [{"message": {"content": "  "}, "finish_reason": "length"}]}
        with self.assertRaises(ValueError):
            parse_response(data)

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            parse_response({"error": "nope"})


class EmergencyNoticeTest(unittest.TestCase):
    def test_red_flag_phrase_detected(self):
        self.assertTrue(needs_emergency_notice("Saya nyeri dada sejak tadi malam"))

    def test_detection_is_case_insensitive(self):
        self.assertTrue(needs_emergency_notice("I have CHEST PAIN"))

    def test_demo_question_not_flagged(self):
        self.assertFalse(needs_emergency_notice(DEMO_QUESTION))


class ChatErrorTest(unittest.TestCase):
    def test_unreachable_endpoint_is_actionable(self):
        def failing_opener(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        with self.assertRaises(RuntimeError) as ctx:
            chat("halo", base_url="http://127.0.0.1:1/v1", opener=failing_opener)
        self.assertIn("start the local server", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
