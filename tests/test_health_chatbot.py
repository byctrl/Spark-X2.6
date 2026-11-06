"""Offline tests for the health-information chatbot (no model download)."""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  Apa itu demam?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Apa itu demam?"})

    def test_history_is_preserved_in_order(self):
        history = [
            {"role": "user", "content": "Saya batuk."},
            {"role": "assistant", "content": "Sejak kapan?"},
        ]
        messages = hc.build_messages("Tiga hari.", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_invalid_history_role_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("halo", [{"role": "system", "content": "x"}])


class PayloadTest(unittest.TestCase):
    def test_payload_is_bounded_and_uses_target_model(self):
        payload = hc.build_payload(hc.build_messages(hc.DEMO_QUESTION))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertLessEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])


class ExtractTextTest(unittest.TestCase):
    def test_reads_assistant_content(self):
        response = {"choices": [{"message": {"content": "  Hubungi klinik.  "}}]}
        self.assertEqual(hc.extract_text(response), "Hubungi klinik.")

    def test_malformed_response_rejected(self):
        with self.assertRaises(ValueError):
            hc.extract_text({"choices": []})

    def test_empty_content_rejected(self):
        with self.assertRaises(ValueError):
            hc.extract_text({"choices": [{"message": {"content": "  "}}]})


class SafetyPromptTest(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_requires_clinician(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("clinician", prompt)

    def test_demo_question_is_non_empty(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


if __name__ == "__main__":
    unittest.main()
