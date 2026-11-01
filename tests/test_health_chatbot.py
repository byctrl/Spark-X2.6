import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import health_chatbot as hc  # noqa: E402


class FakeGenerator:
    tokenizer = None

    def __init__(self, text):
        self.text = text
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return [{"generated_text": self.text}]


class BuildMessagesTest(unittest.TestCase):
    def test_starts_with_system_prompt(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)

    def test_question_is_last_user_turn_and_stripped(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_preserved(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = hc.build_messages("Three days.", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1], history[0])
        self.assertEqual(messages[2], history[1])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_bad_history_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("Hi", [{"role": "system", "content": "nope"}])

    def test_safety_boundary_in_prompt(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        for phrase in ("never diagnose", "never prescribe", "emergency"):
            self.assertIn(phrase, lowered)


class FormatPromptTest(unittest.TestCase):
    def test_fallback_contains_roles_and_generation_prompt(self):
        prompt = hc.format_prompt(hc.build_messages("What is a fever?"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("What is a fever?", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>\n"))


class GenerateReplyTest(unittest.TestCase):
    def test_uses_injected_generator(self):
        generator = FakeGenerator("See a clinician if the fever lasts.")
        reply = hc.generate_reply("I have a fever.", generator=generator)
        self.assertEqual(reply, "See a clinician if the fever lasts.")
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(generator.calls[0]["kwargs"]["max_new_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(generator.calls[0]["kwargs"]["return_full_text"])

    def test_strips_echoed_prompt(self):
        generator = FakeGenerator("")
        prompt = hc.format_prompt(hc.build_messages("I have a fever."))
        generator.text = prompt + "Rest and monitor your temperature."
        reply = hc.generate_reply("I have a fever.", generator=generator)
        self.assertEqual(reply, "Rest and monitor your temperature.")


if __name__ == "__main__":
    unittest.main()
