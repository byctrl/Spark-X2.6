"""Tests for the minimal Spark-X2.6-4B health chatbot demo.

Run from the repository root:
    python -m unittest discover -s src -p 'test_*.py'
"""

import unittest

import health_chatbot as hc


class ConfigTest(unittest.TestCase):
    def test_model_and_budget(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_forbids_diagnosis(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("not diagnose", prompt)
        self.assertIn("not prescribe", prompt)
        self.assertIn("clinician", prompt)

    def test_demo_question_is_usable(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())
        self.assertEqual(hc.build_messages(hc.DEMO_QUESTION)[1]["content"], hc.DEMO_QUESTION)


class BuildMessagesTest(unittest.TestCase):
    def test_system_and_user_roles(self):
        messages = hc.build_messages("hello")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "hello")

    def test_question_is_stripped(self):
        self.assertEqual(hc.build_messages("  hi  ")[1]["content"], "hi")

    def test_empty_question_rejected(self):
        for value in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(value)


class RenderPromptTest(unittest.TestCase):
    def test_fallback_prompt_contains_question_and_assistant_turn(self):
        prompt = hc.render_prompt(hc.build_messages("fever"))
        self.assertIn("fever", prompt)
        self.assertIn(hc.SYSTEM_PROMPT, prompt)
        self.assertTrue(prompt.rstrip().endswith("assistant"))

    def test_tokenizer_chat_template_is_preferred(self):
        class StubTokenizer:
            chat_template = "stub"

            def apply_chat_template(
                self, messages, tokenize=False, add_generation_prompt=True
            ):
                return "TEMPLATED:" + messages[-1]["content"]

        prompt = hc.render_prompt(hc.build_messages("q"), StubTokenizer())
        self.assertEqual(prompt, "TEMPLATED:q")


class DisclaimerTest(unittest.TestCase):
    def test_disclaimer_appended(self):
        result = hc.with_disclaimer("Drink water and rest.")
        self.assertTrue(result.startswith("Drink water and rest."))
        self.assertTrue(result.endswith(hc.DISCLAIMER))

    def test_empty_answer_still_warns(self):
        self.assertEqual(hc.with_disclaimer("   "), hc.DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
