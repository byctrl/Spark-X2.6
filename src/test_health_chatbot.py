"""Tests for the minimal Spark-X2.6-4B health-information chatbot."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class PromptConfigTests(unittest.TestCase):
    def test_model_id_is_spark(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_max_new_tokens(self):
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_mentions_clinician(self):
        self.assertIn("clinician", hc.SYSTEM_PROMPT)

    def test_system_prompt_forbids_diagnosis(self):
        self.assertIn("diagnose", hc.SYSTEM_PROMPT.lower())

    def test_system_prompt_forbids_prescribing(self):
        self.assertIn("prescribe", hc.SYSTEM_PROMPT.lower())

    def test_system_prompt_mentions_urgent_care(self):
        self.assertIn("urgent", hc.SYSTEM_PROMPT.lower())

    def test_demo_question_is_available(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


class BuildMessagesTests(unittest.TestCase):
    def test_messages_start_with_system_prompt(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})

    def test_messages_end_with_user_question(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_kept_before_the_question(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
        ]
        messages = hc.build_messages("What is a fever?", history=history)
        self.assertEqual(
            [m["content"] for m in messages[1:3]],
            ["Hello", "Hi, how can I help?"],
        )

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class AnswerTests(unittest.TestCase):
    def test_answer_uses_injected_generator(self):
        seen = {}

        def fake_generate(prompt, max_new_tokens):
            seen["prompt"] = prompt
            seen["max_new_tokens"] = max_new_tokens
            return "  Please see a clinician.  "

        reply = hc.answer("What is a fever?", generate=fake_generate)
        self.assertEqual(reply, "Please see a clinician.")
        self.assertEqual(seen["max_new_tokens"], hc.MAX_NEW_TOKENS)
        self.assertIn("What is a fever?", seen["prompt"])
        self.assertIn("clinician", seen["prompt"])

    def test_chat_alias_works(self):
        reply = hc.chat("What is a fever?", generate=lambda prompt, n: "ok")
        self.assertEqual(reply, "ok")


if __name__ == "__main__":
    unittest.main()
