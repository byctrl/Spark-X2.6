"""Standard-library tests for the Spark-X2.6-4B health chatbot demo.

Run from the repository root:
    python -m unittest discover -s src -p "test_*.py" -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as chatbot  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_is_first_and_user_turn_is_last(self):
        messages = chatbot.build_messages("  I have a fever.  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], chatbot.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "I have a fever."})

    def test_history_is_preserved_between_system_and_user(self):
        history = [{"role": "user", "content": "hello"},
                   {"role": "assistant", "content": "hi"}]
        messages = chatbot.build_messages("next question", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"], "hello")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[3]["content"], "next question")

    def test_blank_question_is_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                chatbot.build_messages(bad)


class FormatPromptTest(unittest.TestCase):
    def test_prompt_ends_with_assistant_marker(self):
        prompt = chatbot.format_prompt(chatbot.build_messages("question"))
        self.assertTrue(prompt.endswith("<|assistant|>\n"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)


class SafetyBoundaryTest(unittest.TestCase):
    def test_system_prompt_keeps_no_diagnosis_boundary(self):
        lowered = chatbot.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", lowered)
        self.assertIn("clinician", lowered)

    def test_model_defaults(self):
        self.assertEqual(chatbot.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertGreater(chatbot.MAX_NEW_TOKENS, 0)
        self.assertTrue(chatbot.DEMO_QUESTION.strip())


if __name__ == "__main__":
    unittest.main()
