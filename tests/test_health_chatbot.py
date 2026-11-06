"""Standard-library tests for the message/prompt layer of the health chatbot demo.

These tests never download or run Spark-X2.6-4B; they pin the safety-relevant
prompt contract and the chat-message construction the model call depends on.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_demo_question_produces_system_then_user(self):
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": hc.DEMO_QUESTION})

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_order_is_preserved_and_invalid_turns_skipped(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi, how can I help?"},
            {"role": "tool", "content": "should be ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("a question", history)
        self.assertEqual(
            [m["role"] for m in messages], ["system", "user", "assistant", "user"]
        )
        self.assertEqual(messages[-1]["content"], "a question")

    def test_system_prompt_keeps_safety_boundaries(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        for needle in ("diagnose", "prescribe", "clinician"):
            self.assertIn(needle, prompt)

    def test_question_is_whitespace_trimmed(self):
        messages = hc.build_messages("  when should I see a doctor?  ")
        self.assertEqual(messages[-1]["content"], "when should I see a doctor?")


class BuildPromptTest(unittest.TestCase):
    def test_fallback_prompt_contains_roles_and_generation_cue(self):
        prompt = hc.build_prompt(hc.build_messages("q"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertTrue(prompt.rstrip().endswith("<|assistant|>"))

    def test_tokenizer_chat_template_is_preferred(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                self.asserted = (tokenize, add_generation_prompt)
                return "TEMPLATED:" + messages[-1]["content"]

        tokenizer = FakeTokenizer()
        prompt = hc.build_prompt(hc.build_messages("q"), tokenizer)
        self.assertEqual(prompt, "TEMPLATED:q")
        self.assertEqual(tokenizer.asserted, (False, True))

    def test_broken_chat_template_falls_back(self):
        class BrokenTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise RuntimeError("no chat template")

        prompt = hc.build_prompt(hc.build_messages("q"), BrokenTokenizer())
        self.assertIn("<|system|>", prompt)


if __name__ == "__main__":
    unittest.main()
