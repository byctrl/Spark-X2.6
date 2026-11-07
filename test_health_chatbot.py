"""Tests for the minimal Spark-X2.6-4B health-information chatbot."""

import unittest

from health_chatbot import (
    DISCLAIMER,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    answer,
    build_messages,
    format_prompt,
    render_prompt,
)


class BuildMessagesTest(unittest.TestCase):
    def test_messages_include_system_and_user(self):
        messages = build_messages("  Apa itu demam?  ")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "Apa itu demam?")

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("   ")

    def test_system_prompt_is_safety_bounded(self):
        lowered = SYSTEM_PROMPT.lower()
        for word in ("diagnos", "prescri", "clinician", "emergency"):
            self.assertIn(word, lowered)

    def test_model_constants(self):
        self.assertEqual(MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(MAX_NEW_TOKENS, 1024)


class PromptTest(unittest.TestCase):
    def test_format_prompt_chatml(self):
        prompt = format_prompt(build_messages("Halo"))
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_render_prompt_prefers_chat_template(self):
        class FakeTokenizer:
            def apply_chat_template(
                self, messages, tokenize=False, add_generation_prompt=True
            ):
                self.called = True
                return "TEMPLATED:" + messages[-1]["content"]

        tokenizer = FakeTokenizer()
        rendered = render_prompt(tokenizer, build_messages("Halo"))
        self.assertEqual(rendered, "TEMPLATED:Halo")
        self.assertTrue(tokenizer.called)

    def test_render_prompt_falls_back(self):
        rendered = render_prompt(object(), build_messages("Halo"))
        self.assertIn("Halo", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))


class AnswerTest(unittest.TestCase):
    def test_answer_uses_injected_generator_and_adds_disclaimer(self):
        captured = {}

        def fake_generate(messages, max_new_tokens):
            captured["messages"] = messages
            captured["max_new_tokens"] = max_new_tokens
            return "  Demam adalah respons tubuh terhadap infeksi.  "

        reply = answer("Apa itu demam?", generate_fn=fake_generate)
        self.assertTrue(reply.startswith("Demam adalah respons tubuh"))
        self.assertTrue(reply.endswith(DISCLAIMER))
        self.assertEqual(captured["max_new_tokens"], MAX_NEW_TOKENS)
        self.assertEqual(captured["messages"][1]["content"], "Apa itu demam?")

    def test_empty_model_output_raises(self):
        with self.assertRaises(RuntimeError):
            answer("Apa itu demam?", generate_fn=lambda messages, n: "   ")

    def test_empty_question_raises_before_generation(self):
        called = {"value": False}

        def fake_generate(messages, max_new_tokens):
            called["value"] = True
            return "text"

        with self.assertRaises(ValueError):
            answer("", generate_fn=fake_generate)
        self.assertFalse(called["value"])


if __name__ == "__main__":
    unittest.main()
