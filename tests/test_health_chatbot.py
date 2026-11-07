"""Offline tests for the health-information chatbot demo.

No model download and no torch/transformers needed: a fake generator is
injected into answer().
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    ),
)

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_structure(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "What is a fever?")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)

    def test_strips_whitespace(self):
        messages = hc.build_messages("   hello   ")
        self.assertEqual(messages[1]["content"], "hello")

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class FormatPromptTest(unittest.TestCase):
    def test_roles_and_assistant_turn(self):
        prompt = hc.format_prompt(hc.build_messages("hi"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>\n"))


class AnswerTest(unittest.TestCase):
    def test_appends_disclaimer(self):
        captured = {}

        def fake_generator(prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return [{"generated_text": "A fever is a raised body temperature."}]

        result = hc.answer("What is a fever?", generator=fake_generator)
        self.assertIn("A fever is a raised body temperature.", result)
        self.assertIn(hc.DISCLAIMER, result)
        self.assertIn("What is a fever?", captured["prompt"])
        self.assertEqual(captured["kwargs"]["max_new_tokens"], hc.MAX_NEW_TOKENS)

    def test_prompt_echo_removed(self):
        def echo_generator(prompt, **kwargs):
            return [{"generated_text": prompt + "Drink water."}]

        result = hc.answer("What is a fever?", generator=echo_generator)
        self.assertIn("Drink water.", result)
        self.assertNotIn("<|assistant|>", result)

    def test_disclaimer_not_duplicated(self):
        def fake_generator(prompt, **kwargs):
            return [{"generated_text": "See a clinician. " + hc.DISCLAIMER}]

        result = hc.answer("hi", generator=fake_generator)
        self.assertEqual(result.count(hc.DISCLAIMER), 1)

    def test_empty_output_falls_back(self):
        def empty_generator(prompt, **kwargs):
            return [{"generated_text": ""}]

        result = hc.answer("hi", generator=empty_generator)
        self.assertIn(hc.DISCLAIMER, result)
        self.assertTrue(result.strip())


class PromptSafetyTest(unittest.TestCase):
    def test_system_prompt_forbids_diagnosis(self):
        text = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", text)
        self.assertIn("emergency", text)
        self.assertIn("clinician", text)

    def test_model_and_demo_constants(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertTrue(hc.DEMO_QUESTION.strip())
        self.assertTrue(hc.DISCLAIMER.strip())


if __name__ == "__main__":
    unittest.main()
