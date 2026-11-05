"""Unit tests for the Spark-X2.6-4B health-information chatbot demo."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_includes_system_prompt_and_question(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "What is a fever?")

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class AnswerTest(unittest.TestCase):
    def test_uses_model_reply_and_adds_disclaimer(self):
        seen = {}

        def fake_complete(messages):
            seen["messages"] = messages
            return "  Rest and drink water.  "

        reply = hc.answer("I have a cough", complete_fn=fake_complete)
        self.assertTrue(reply.startswith("Rest and drink water."))
        self.assertIn(hc.DISCLAIMER, reply)
        self.assertEqual(seen["messages"][1]["content"], "I have a cough")

    def test_disclaimer_is_not_duplicated(self):
        text = "Stay hydrated.\n\n" + hc.DISCLAIMER
        reply = hc.answer("cough", complete_fn=lambda messages: text)
        self.assertEqual(reply.count(hc.DISCLAIMER), 1)

    def test_empty_reply_is_an_error(self):
        with self.assertRaises(RuntimeError):
            hc.answer("cough", complete_fn=lambda messages: "   ")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.answer("", complete_fn=lambda messages: "irrelevant")


class PromptSafetyTest(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_dosage(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("dosage", prompt)
        self.assertIn("clinician", prompt)

    def test_demo_question_and_model_id_are_set(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertTrue(hc.DEMO_QUESTION.strip())


if __name__ == "__main__":
    unittest.main()
