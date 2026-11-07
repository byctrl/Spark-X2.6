"""Tests for the Spark-X2.6-4B health chatbot (no model weights needed)."""

import unittest

from health_chatbot import (
    EMERGENCY_NOTE,
    EMPTY_REPLY_NOTE,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    answer_question,
    build_messages,
    build_prompt,
    detect_emergency,
)


class FakeGenerator:
    def __init__(self, reply="See a clinician about your cough and fever."):
        self.reply = reply
        self.calls = []

    def __call__(self, prompt, max_new_tokens=MAX_NEW_TOKENS):
        self.calls.append((prompt, max_new_tokens))
        return self.reply


class BuildMessagesTest(unittest.TestCase):
    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            build_messages("   ")

    def test_system_prompt_blocks_diagnosis(self):
        messages = build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("NOT diagnose", messages[0]["content"])
        self.assertIn("qualified health worker", messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_extra_context_is_added_before_question(self):
        messages = build_messages("What is this?", extra_context="Clinic opens at 8am.")
        self.assertEqual(len(messages), 3)
        self.assertIn("Clinic opens at 8am.", messages[1]["content"])
        self.assertEqual(messages[-1]["role"], "user")

    def test_empty_context_is_ignored(self):
        messages = build_messages("What is this?", extra_context="   ")
        self.assertEqual(len(messages), 2)


class BuildPromptTest(unittest.TestCase):
    def test_fallback_prompt_contains_roles_and_question(self):
        prompt = build_prompt(build_messages("Why am I tired?"))
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("Why am I tired?", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_uses_tokenizer_chat_template_when_available(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "TEMPLATED:" + messages[-1]["content"]

        prompt = build_prompt(build_messages("Hello"), tokenizer=Tokenizer())
        self.assertEqual(prompt, "TEMPLATED:Hello")

    def test_broken_template_falls_back(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise RuntimeError("no template")

        prompt = build_prompt(build_messages("Hello"), tokenizer=Tokenizer())
        self.assertIn("Hello", prompt)


class EmergencyScreenTest(unittest.TestCase):
    def test_detects_emergency_wording(self):
        self.assertTrue(detect_emergency("I have chest pain right now"))
        self.assertTrue(detect_emergency("my father cannot breathe"))

    def test_ignores_ordinary_question(self):
        self.assertFalse(detect_emergency("How much water should I drink?"))


class AnswerQuestionTest(unittest.TestCase):
    def test_returns_reply_and_metadata(self):
        generator = FakeGenerator("A fever is a higher body temperature.")
        result = answer_question("What is a fever?", generator)
        self.assertEqual(result["model"], MODEL_ID)
        self.assertEqual(result["question"], "What is a fever?")
        self.assertFalse(result["emergency"])
        self.assertEqual(result["answer"], "A fever is a higher body temperature.")
        self.assertEqual(len(generator.calls), 1)
        self.assertIn(SYSTEM_PROMPT, generator.calls[0][0])
        self.assertEqual(generator.calls[0][1], MAX_NEW_TOKENS)

    def test_emergency_note_is_prefixed(self):
        result = answer_question("I have chest pain", FakeGenerator("See a doctor."))
        self.assertTrue(result["emergency"])
        self.assertTrue(result["answer"].startswith(EMERGENCY_NOTE))

    def test_emergency_note_not_duplicated(self):
        result = answer_question("I have chest pain", FakeGenerator(EMERGENCY_NOTE))
        self.assertEqual(result["answer"].count(EMERGENCY_NOTE), 1)

    def test_empty_reply_is_replaced(self):
        result = answer_question("What is a fever?", FakeGenerator("  "))
        self.assertEqual(result["answer"], EMPTY_REPLY_NOTE)

    def test_max_new_tokens_is_forwarded(self):
        generator = FakeGenerator()
        answer_question("What is a fever?", generator, max_new_tokens=64)
        self.assertEqual(generator.calls[0][1], 64)


if __name__ == "__main__":
    unittest.main()
