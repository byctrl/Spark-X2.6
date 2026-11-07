"""Unit tests for the Spark-X2.6-4B health chatbot (no model download)."""

import unittest

import health_chatbot as hc


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.calls.append((messages, tokenize, add_generation_prompt))
        return "PROMPT::" + "|".join(m["content"] for m in messages)


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_question_is_trimmed(self):
        messages = hc.build_messages("  hello  ")
        self.assertEqual(messages[-1]["content"], "hello")

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        messages = hc.build_messages("second", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "second")

    def test_empty_question_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_invalid_history_turn_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hi", [{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("hi", [{"role": "user", "content": "  "}])


class ConstantsTest(unittest.TestCase):
    def test_model_and_budget(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_safety_boundaries_in_system_prompt(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("not medical advice", prompt)
        self.assertIn("do not diagnose", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("emergency", prompt)

    def test_demo_question_is_present(self):
        self.assertIn("fever", hc.DEMO_QUESTION.lower())


class BuildPromptTest(unittest.TestCase):
    def test_uses_chat_template_when_available(self):
        tokenizer = FakeTokenizer()
        messages = hc.build_messages("hello")
        prompt = hc.build_prompt(messages, tokenizer)
        self.assertTrue(prompt.startswith("PROMPT::"))
        self.assertEqual(tokenizer.calls[0][1], False)
        self.assertEqual(tokenizer.calls[0][2], True)

    def test_fallback_prompt_has_roles(self):
        prompt = hc.build_prompt(hc.build_messages("hello"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>"))


class DisclaimerTest(unittest.TestCase):
    def test_disclaimer_appended_once(self):
        text = hc.with_disclaimer("  Take care.  ")
        self.assertTrue(text.startswith("Take care."))
        self.assertTrue(text.endswith(hc.DISCLAIMER))

    def test_empty_answer_still_gets_disclaimer(self):
        self.assertEqual(hc.with_disclaimer(""), hc.DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
