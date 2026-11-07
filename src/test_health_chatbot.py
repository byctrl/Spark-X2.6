"""Offline unit tests for the Spark-X2.6-4B health-information chatbot."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("Is a fever normal after travel?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(
            messages[-1],
            {"role": "user", "content": "Is a fever normal after travel?"},
        )

    def test_question_is_stripped(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[-1]["content"], "What is a fever?")

    def test_empty_or_invalid_question_rejected(self):
        for bad in ("   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_history_kept_and_sanitised(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
            "not-a-turn",
        ]
        messages = hc.build_messages("Two weeks.", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[-1]["content"], "Two weeks.")

    def test_system_prompt_blocks_diagnosis(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("not medical advice", prompt)


class PayloadTest(unittest.TestCase):
    def test_payload_targets_spark_model(self):
        payload = hc.build_payload(hc.build_messages("hello"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][-1]["content"], "hello")


class ExtractAnswerTest(unittest.TestCase):
    def test_reads_content(self):
        data = {"choices": [{"message": {"content": "  See a clinician.  "}}]}
        self.assertEqual(hc.extract_answer(data), "See a clinician.")

    def test_rejects_malformed_responses(self):
        for data in (
            None,
            {},
            {"choices": []},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": "   "}}]},
        ):
            with self.assertRaises(ValueError):
                hc.extract_answer(data)


class ChatTest(unittest.TestCase):
    def test_chat_posts_to_openai_compatible_endpoint(self):
        seen = {}

        def transport(url, payload):
            seen["url"] = url
            seen["payload"] = payload
            return {"choices": [{"message": {"content": "General education only."}}]}

        answer = hc.chat(
            "What is a fever?",
            base_url="http://127.0.0.1:30000/v1/",
            transport=transport,
        )
        self.assertEqual(answer, "General education only.")
        self.assertEqual(seen["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(seen["payload"]["model"], hc.MODEL_ID)

    def test_chat_propagates_bad_response(self):
        with self.assertRaises(ValueError):
            hc.chat("hi", transport=lambda url, payload: {})


if __name__ == "__main__":
    unittest.main()
