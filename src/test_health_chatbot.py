"""Unit tests for src/health_chatbot.py (no model download required)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class FakeGenerator:
    """Stands in for a transformers text-generation pipeline."""

    tokenizer = None

    def __init__(self, text):
        self.text = text
        self.prompts = []
        self.kwargs = []

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        return [{"generated_text": self.text}]


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("I have a fever.")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(
            messages[-1], {"role": "user", "content": "I have a fever."}
        )

    def test_history_kept_and_junk_dropped(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("next", history)
        roles = [message["role"] for message in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_system_prompt_forbids_diagnosis(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never give a diagnosis", lowered)
        self.assertIn("clinician", lowered)


class RenderPromptTest(unittest.TestCase):
    def test_fallback_is_readable(self):
        prompt = hc.render_prompt(hc.build_messages("cough"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertIn("<|assistant|>", prompt)

    def test_uses_chat_template_when_available(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize,
                                    add_generation_prompt):
                return "TEMPLATED"

        self.assertEqual(
            hc.render_prompt(hc.build_messages("cough"), Tokenizer()),
            "TEMPLATED",
        )


class FinalizeReplyTest(unittest.TestCase):
    def test_empty_reply_still_safe(self):
        reply = hc.finalize_reply("   ")
        self.assertIn(hc.DISCLAIMER, reply)

    def test_disclaimer_added_once(self):
        once = hc.finalize_reply("Rest and drink water.")
        twice = hc.finalize_reply(once)
        self.assertEqual(once, twice)
        self.assertEqual(once.lower().count("not medical advice"), 1)


class GenerateReplyTest(unittest.TestCase):
    def test_prompt_and_output_are_handled(self):
        generator = FakeGenerator("A fever after travel is worth checking.")
        reply = hc.generate_reply("I have a fever.", generator=generator)

        self.assertEqual(len(generator.prompts), 1)
        self.assertIn("I have a fever.", generator.prompts[0])
        self.assertEqual(
            generator.kwargs[0]["max_new_tokens"], hc.MAX_NEW_TOKENS
        )
        self.assertTrue(reply.startswith("A fever after travel"))
        self.assertIn(hc.DISCLAIMER, reply)

    def test_nested_pipeline_output_is_handled(self):
        class NestedGenerator:
            tokenizer = None

            def __call__(self, prompt, **kwargs):
                return [[{"generated_text": "Nested answer."}]]

        reply = hc.generate_reply("q", generator=NestedGenerator())
        self.assertTrue(reply.startswith("Nested answer."))

    def test_bad_token_budget_rejected(self):
        with self.assertRaises(ValueError):
            hc.generate_reply(
                "q", generator=FakeGenerator("x"), max_new_tokens=0
            )


if __name__ == "__main__":
    unittest.main()
