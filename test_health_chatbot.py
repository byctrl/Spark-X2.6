"""Tests for health_chatbot.py.

Run from the repository root:

    python -m unittest test_health_chatbot -v

No model, network access or extra dependency is required.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as chatbot


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_and_question_are_present(self):
        messages = chatbot.build_messages("What is a fever?")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("Never diagnose", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "What is a fever?")

    def test_question_is_stripped(self):
        messages = chatbot.build_messages("  cough  ")
        self.assertEqual(messages[1]["content"], "cough")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            chatbot.build_messages("   ")


class PayloadTest(unittest.TestCase):
    def test_payload_targets_spark_model(self):
        payload = chatbot.build_payload("hello")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])
        self.assertEqual(len(payload["messages"]), 2)


class ExtractTextTest(unittest.TestCase):
    def test_reads_first_choice(self):
        response = {"choices": [{"message": {"content": "  Drink water.  "}}]}
        self.assertEqual(chatbot.extract_text(response), "Drink water.")

    def test_rejects_empty_answer(self):
        with self.assertRaises(ValueError):
            chatbot.extract_text({"choices": [{"message": {"content": "   "}}]})

    def test_rejects_malformed_response(self):
        with self.assertRaises(ValueError):
            chatbot.extract_text({"error": "boom"})


class GuardTest(unittest.TestCase):
    SAFE = (
        "A fever is a common sign of infection. A clinician can check what is "
        "causing yours before anything else is decided."
    )

    def test_safe_answer_has_no_warnings(self):
        self.assertEqual(chatbot.check_answer(self.SAFE), [])

    def test_diagnosis_claim_is_flagged(self):
        self.assertTrue(chatbot.check_answer("Your diagnosis is malaria."))

    def test_prescription_claim_is_flagged(self):
        self.assertTrue(chatbot.check_answer("You should take amoxicillin tonight."))

    def test_disclaimer_is_always_appended(self):
        text, warnings = chatbot.compose_answer(self.SAFE, "what is a fever")
        self.assertTrue(text.endswith(chatbot.DISCLAIMER))
        self.assertEqual(warnings, [])

    def test_red_flag_adds_emergency_note(self):
        text, _ = chatbot.compose_answer(self.SAFE, "I have chest pain since this morning")
        self.assertTrue(text.startswith(chatbot.EMERGENCY_NOTE))


class AskTest(unittest.TestCase):
    def setUp(self):
        self.calls = []

    def transport(self, url, payload, timeout=None):
        self.calls.append((url, payload))
        return {
            "choices": [
                {"message": {"content": "A fever is common. A clinician can check it."}}
            ]
        }

    def test_ask_calls_openai_compatible_endpoint(self):
        result = chatbot.ask(
            "what is a fever?",
            transport=self.transport,
            base_url="http://127.0.0.1:30000/v1/",
        )
        self.assertEqual(self.calls[0][0], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(self.calls[0][1]["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(result["model"], "XHToken/Spark-X2.6-4B")
        self.assertIn("A fever is common.", result["answer"])
        self.assertTrue(result["answer"].endswith(chatbot.DISCLAIMER))
        self.assertEqual(result["warnings"], [])

    def test_ask_red_flag_question_gets_emergency_note(self):
        result = chatbot.ask(
            "my father has chest pain",
            transport=self.transport,
            base_url="http://127.0.0.1:30000/v1",
        )
        self.assertTrue(result["answer"].startswith(chatbot.EMERGENCY_NOTE))

    def test_ask_requires_question(self):
        with self.assertRaises(ValueError):
            chatbot.ask("  ", transport=self.transport)


if __name__ == "__main__":
    unittest.main()
