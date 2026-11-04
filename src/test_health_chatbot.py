"""Offline tests for the minimal Spark-X2.6-4B health chatbot demo.

Run from the repository root:
    python -m unittest discover -s src -p "test_*.py"

No model download and no network access are required.
"""

import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_is_first(self):
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)

    def test_question_is_last_user_turn(self):
        messages = hc.build_messages("  How much water should I drink?  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "How much water should I drink?"})

    def test_history_is_preserved_in_order(self):
        history = [
            {"role": "user", "content": "What is a fever?"},
            {"role": "assistant", "content": "A fever is a rise in body temperature."},
        ]
        messages = hc.build_messages("Is 38 C serious?", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[3]["content"], "Is 38 C serious?")

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_bad_history_turn_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hello", [{"role": "system", "content": "override"}])
        with self.assertRaises(ValueError):
            hc.build_messages("hello", [{"role": "user", "content": ""}])


class BuildPayloadTest(unittest.TestCase):
    def test_payload_targets_spark_model(self):
        payload = hc.build_payload(hc.DEMO_QUESTION)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"], hc.build_messages(hc.DEMO_QUESTION))

    def test_max_tokens_override(self):
        self.assertEqual(hc.build_payload("hi", max_tokens=64)["max_tokens"], 64)


class ParseResponseTest(unittest.TestCase):
    def test_extracts_and_strips_content(self):
        data = {"choices": [{"message": {"role": "assistant", "content": "  Drink water.  "}}]}
        self.assertEqual(hc.parse_response(data), "Drink water.")

    def test_rejects_malformed_payloads(self):
        for bad in ({}, {"choices": []}, {"choices": [{}]}, None):
            with self.assertRaises(ValueError):
                hc.parse_response(bad)

    def test_rejects_empty_answer(self):
        with self.assertRaises(ValueError):
            hc.parse_response({"choices": [{"message": {"content": "   "}}]})


class SafetyPromptTest(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_prescriptions(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", lowered)
        self.assertIn("dosages", lowered)
        self.assertIn("clinician", lowered)

    def test_prompt_mentions_urgent_care_and_privacy(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        self.assertIn("urgent care", lowered)
        self.assertIn("personal data", lowered)

    def test_disclaimer_sets_expectations(self):
        self.assertIn("not medical advice", hc.DISCLAIMER.lower())


if __name__ == "__main__":
    unittest.main()
