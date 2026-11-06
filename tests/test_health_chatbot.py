"""Tests for the Spark-X2.6-4B health-information chatbot demo."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("must NOT diagnose", messages[0]["content"])
        self.assertEqual(
            messages[-1], {"role": "user", "content": "What is a fever?"}
        )

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_order_and_input_truncation(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "dropped"},
            {"role": "assistant", "content": "hello"},
        ]
        messages = hc.build_messages("x" * (hc.MAX_INPUT_CHARS + 50), history=history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(len(messages[-1]["content"]), hc.MAX_INPUT_CHARS)

    def test_prompt_contains_role_markers(self):
        prompt = hc.build_prompt(hc.build_messages("hello"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>\nhello\n", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>\n"))


class AnswerTest(unittest.TestCase):
    def test_injected_generator_and_safety_notice(self):
        seen = {}

        def fake_generate(messages, max_new_tokens=0):
            seen["messages"] = messages
            seen["max_new_tokens"] = max_new_tokens
            return "  Rest and watch for worsening symptoms.  "

        reply = hc.answer(hc.DEMO_QUESTION, generate=fake_generate)
        self.assertTrue(reply.startswith("Rest and watch"))
        self.assertIn(hc.SAFETY_NOTICE, reply)
        self.assertEqual(seen["messages"][-1]["content"], hc.DEMO_QUESTION)
        self.assertEqual(seen["max_new_tokens"], hc.MAX_NEW_TOKENS)

    def test_empty_model_output_gets_fallback_and_notice(self):
        reply = hc.answer("hello", generate=lambda messages, max_new_tokens=0: "")
        self.assertIn("could not produce an answer", reply)
        self.assertIn(hc.SAFETY_NOTICE, reply)

    def test_safety_notice_not_duplicated(self):
        reply = hc.answer(
            "hello",
            generate=lambda messages, max_new_tokens=0: "ok\n\n" + hc.SAFETY_NOTICE,
        )
        self.assertEqual(reply.count(hc.SAFETY_NOTICE), 1)


if __name__ == "__main__":
    unittest.main()
