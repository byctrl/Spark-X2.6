#!/usr/bin/env python3
"""Offline unit tests for the Spark-X2.6-4B health chatbot (no network)."""

import unittest

import health_chatbot as hc


class ValidateQuestionTest(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(hc.validate_question("  a \n  b  "), "a b")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            hc.validate_question("   ")

    def test_rejects_non_string(self):
        with self.assertRaises(TypeError):
            hc.validate_question(None)

    def test_truncates_long_questions(self):
        self.assertEqual(len(hc.validate_question("x" * 5000)), hc.MAX_QUESTION_CHARS)


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("How much water should I drink?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(
            messages[1],
            {"role": "user", "content": "How much water should I drink?"},
        )

    def test_prompt_forbids_diagnosis_and_names_clinician(self):
        self.assertIn("Never diagnose", hc.SYSTEM_PROMPT)
        self.assertIn("clinician", hc.SYSTEM_PROMPT.lower())


class PostprocessTest(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(hc.postprocess("  hello  "), "hello")

    def test_empty_reply_keeps_safety_note(self):
        self.assertIn("not a diagnosis", hc.postprocess(None))


class ExtractReplyTest(unittest.TestCase):
    def test_reads_openai_shape(self):
        body = {"choices": [{"message": {"content": "rest and fluids"}}]}
        self.assertEqual(hc.extract_reply(body), "rest and fluids")

    def test_rejects_unknown_shape(self):
        with self.assertRaises(ValueError):
            hc.extract_reply({"error": "boom"})


class ConfigTest(unittest.TestCase):
    def test_demo_constants(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)
        self.assertTrue(hc.DEMO_QUESTION.strip())


if __name__ == "__main__":
    unittest.main()
