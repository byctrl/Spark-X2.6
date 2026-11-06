"""Offline tests for src/health_chatbot.py (no network, no model needed).

Run from the repository root:
    python -m unittest discover -s src -p "test_*.py"
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("What foods are rich in iron?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)

    def test_question_is_last_user_turn(self):
        messages = hc.build_messages("What foods are rich in iron?")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "What foods are rich in iron?")

    def test_prompt_forbids_diagnosis_and_prescription(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("not diagnose", prompt)
        self.assertIn("not prescribe", prompt)
        self.assertIn("not medical advice", prompt)

    def test_history_is_copied_and_filtered(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("next question", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(len(history), 4)


class PayloadTest(unittest.TestCase):
    def test_payload_targets_spark_model(self):
        payload = hc.build_payload("Is a mild headache normal?")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "system")

    def test_max_tokens_can_be_overridden(self):
        payload = hc.build_payload("q", max_tokens=64)
        self.assertEqual(payload["max_tokens"], 64)


class ExtractTextTest(unittest.TestCase):
    def test_plain_string_content(self):
        body = {"choices": [{"message": {"role": "assistant", "content": " Hello "}}]}
        self.assertEqual(hc.extract_text(body), "Hello")

    def test_list_content_parts(self):
        body = {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}
        self.assertEqual(hc.extract_text(body), "ab")

    def test_missing_content_is_empty(self):
        self.assertEqual(hc.extract_text({}), "")
        self.assertEqual(hc.extract_text(None), "")
        self.assertEqual(hc.extract_text({"choices": []}), "")


class SafetyTest(unittest.TestCase):
    def test_emergency_wording_is_detected(self):
        self.assertTrue(hc.looks_like_emergency("I have chest pain and trouble breathing"))
        self.assertTrue(hc.looks_like_emergency("I want to kill myself"))

    def test_ordinary_question_is_not_emergency(self):
        self.assertFalse(hc.looks_like_emergency("Which vegetables have iron?"))

    def test_notice_is_prepended_for_emergency(self):
        answer = hc.apply_safety_notice("I have chest pain", "Please rest.")
        self.assertTrue(answer.startswith(hc.EMERGENCY_NOTICE))
        self.assertTrue(answer.endswith("Please rest."))

    def test_answer_untouched_without_emergency(self):
        self.assertEqual(hc.apply_safety_notice("What is iron?", " A mineral. "), "A mineral.")

    def test_referral_detection(self):
        self.assertTrue(hc.has_referral("Please talk to a doctor about this."))
        self.assertFalse(hc.has_referral("Iron is a mineral."))


if __name__ == "__main__":
    unittest.main()
