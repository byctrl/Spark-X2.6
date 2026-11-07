"""Offline unit tests for the Spark-X2.6 health chatbot demo."""

import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_starts_with_system_prompt(self):
        messages = hc.build_messages("Apa itu demam?")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})

    def test_last_message_is_the_trimmed_question(self):
        messages = hc.build_messages("  How much water should an adult drink?  ")
        self.assertEqual(
            messages[-1],
            {"role": "user", "content": "How much water should an adult drink?"},
        )

    def test_history_kept_in_order_and_filtered(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignore me"},
            {"role": "assistant", "content": "   "},
        ]
        messages = hc.build_messages("next", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")
        with self.assertRaises(ValueError):
            hc.build_messages(None)


class PayloadTest(unittest.TestCase):
    def test_payload_targets_spark_x26_4b(self):
        payload = hc.build_payload(hc.build_messages("question"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])

    def test_system_prompt_states_safety_limits(self):
        text = hc.SYSTEM_PROMPT.lower()
        self.assertIn("diagnos", text)
        self.assertIn("clinician", text)
        self.assertIn("emergency", text)


class ParseResponseTest(unittest.TestCase):
    def test_reads_and_trims_message_content(self):
        data = {
            "choices": [
                {"message": {"role": "assistant", "content": "  Drink water.  "}}
            ]
        }
        self.assertEqual(hc.parse_response(data), "Drink water.")

    def test_rejects_malformed_payloads(self):
        for bad in ({}, {"choices": []}, {"choices": [{"message": {}}]}, None):
            with self.assertRaises(ValueError):
                hc.parse_response(bad)

    def test_rejects_empty_answer(self):
        data = {"choices": [{"message": {"content": "   "}}]}
        with self.assertRaises(ValueError):
            hc.parse_response(data)


if __name__ == "__main__":
    unittest.main()
