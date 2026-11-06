import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(
            messages[-1], {"role": "user", "content": hc.DEMO_QUESTION}
        )

    def test_question_is_stripped(self):
        messages = hc.build_messages("  when to see a doctor?  ")
        self.assertEqual(messages[-1]["content"], "when to see a doctor?")

    def test_empty_question_rejected(self):
        for value in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(value)

    def test_history_kept_in_order(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        messages = hc.build_messages("q", history=history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_bad_history_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("q", history=[{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("q", history=[{"role": "user", "content": "  "}])

    def test_safety_rules_in_system_prompt(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("never recommend prescription drugs", prompt)
        self.assertIn("professional help", prompt)

    def test_format_prompt_ends_with_assistant_turn(self):
        text = hc.format_prompt(hc.build_messages("q"))
        self.assertTrue(text.rstrip().endswith("<|assistant|>"))
        self.assertIn("<|system|>", text)

    def test_answer_uses_injected_generator(self):
        seen = {}

        def fake_generator(messages):
            seen["messages"] = messages
            return "  see a clinician  "

        self.assertEqual(hc.answer("q", generator=fake_generator), "see a clinician")
        self.assertEqual(seen["messages"][-1]["content"], "q")

    def test_answer_handles_empty_output(self):
        self.assertEqual(hc.answer("q", generator=lambda messages: None), "")

    def test_constants(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)


if __name__ == "__main__":
    unittest.main()
