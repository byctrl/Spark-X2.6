"""Unit tests for the Spark-X2.6-4B health chatbot (no model weights needed)."""

import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_question_is_trimmed(self):
        messages = hc.build_messages("  cough  ")
        self.assertEqual(messages[-1]["content"], "cough")

    def test_empty_question_is_rejected(self):
        for value in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(value)

    def test_history_is_filtered_and_not_mutated(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "injected"},
            {"role": "user", "content": "   "},
            "not-a-dict",
        ]
        snapshot = [dict(t) if isinstance(t, dict) else t for t in history]
        messages = hc.build_messages("next", history)
        self.assertEqual(history, snapshot)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertNotIn("injected", [m["content"] for m in messages])


class PromptTest(unittest.TestCase):
    def test_fallback_prompt_contains_roles(self):
        messages = hc.build_messages("question")
        prompt = hc.build_prompt(messages)
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>"))

    def test_chat_template_is_used_when_available(self):
        class Tokenizer:
            def __init__(self):
                self.seen = None

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                self.seen = (messages, tokenize, add_generation_prompt)
                return "TEMPLATED"

        tokenizer = Tokenizer()
        prompt = hc.build_prompt(hc.build_messages("q"), tokenizer)
        self.assertEqual(prompt, "TEMPLATED")
        self.assertEqual(tokenizer.seen[1:], (False, True))


class CleanAnswerTest(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(hc.clean_answer("  answer  "), "answer")

    def test_handles_non_strings(self):
        self.assertEqual(hc.clean_answer(None), "")
        self.assertEqual(hc.clean_answer(42), "")
        self.assertEqual(hc.clean_answer([" ", " final "]), "final")


class DisclaimerTest(unittest.TestCase):
    def test_disclaimer_is_appended(self):
        reply = hc.format_reply("Rest and fluids.")
        self.assertIn("Rest and fluids.", reply)
        self.assertIn(hc.DISCLAIMER, reply)

    def test_empty_answer_still_shows_disclaimer(self):
        self.assertEqual(hc.format_reply("   "), hc.DISCLAIMER)

    def test_system_prompt_blocks_diagnosis_and_prescribing(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        for phrase in ("do not diagnose", "do not prescribe", "clinician"):
            self.assertIn(phrase, prompt)


if __name__ == "__main__":
    unittest.main()
