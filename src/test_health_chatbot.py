"""Offline unit tests for src/health_chatbot.py (no model, no network)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as chatbot  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = chatbot.build_messages("I have a cough.")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], chatbot.SYSTEM_PROMPT)

    def test_question_is_last_user_turn(self):
        messages = chatbot.build_messages("  I have a cough.  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "I have a cough."})

    def test_history_is_preserved_in_order(self):
        history = [
            {"role": "user", "content": "I feel tired."},
            {"role": "assistant", "content": "How long has this lasted?"},
        ]
        messages = chatbot.build_messages("Two weeks.", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"], "I feel tired.")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[3]["content"], "Two weeks.")

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                chatbot.build_messages(bad)

    def test_bad_history_role_is_rejected(self):
        with self.assertRaises(ValueError):
            chatbot.build_messages("Hello", [{"role": "system", "content": "x"}])

    def test_bad_history_content_is_rejected(self):
        with self.assertRaises(ValueError):
            chatbot.build_messages("Hello", [{"role": "user", "content": " "}])


class SafetyPromptTest(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_prescription(self):
        prompt = chatbot.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("prescribe", prompt)

    def test_prompt_routes_user_to_clinician(self):
        self.assertIn("clinician", chatbot.SYSTEM_PROMPT.lower())

    def test_disclaimer_mentions_not_medical_advice(self):
        self.assertIn("not medical advice", chatbot.DISCLAIMER.lower())

    def test_demo_question_is_present(self):
        self.assertTrue(chatbot.DEMO_QUESTION.strip())


class PayloadTest(unittest.TestCase):
    def test_payload_uses_model_id_and_token_budget(self):
        messages = chatbot.build_messages("Hello")
        payload = chatbot.build_payload(messages)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], chatbot.MAX_NEW_TOKENS)
        self.assertEqual(payload["messages"], messages)

    def test_empty_message_list_is_rejected(self):
        with self.assertRaises(ValueError):
            chatbot.build_payload([])


class ExtractTextTest(unittest.TestCase):
    def test_extracts_and_strips_content(self):
        response = {"choices": [{"message": {"content": "  See a clinician.  "}}]}
        self.assertEqual(chatbot.extract_text(response), "See a clinician.")

    def test_missing_choices_is_rejected(self):
        with self.assertRaises(ValueError):
            chatbot.extract_text({})

    def test_empty_content_is_rejected(self):
        response = {"choices": [{"message": {"content": "   "}}]}
        with self.assertRaises(ValueError):
            chatbot.extract_text(response)


if __name__ == "__main__":
    unittest.main()
