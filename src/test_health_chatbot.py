"""Tests for the Spark-X2.6-4B health-information chatbot.

These tests never download the model: a fake generator is injected, so
``python -m unittest discover -s src -p "test_*.py"`` runs anywhere.
"""

import unittest

try:
    from . import health_chatbot
except ImportError:  # running the file / discover without a package
    import health_chatbot


class FakeGenerator:
    def __init__(self, answer="Track the dates of the fever and your travel history."):
        self.answer = answer
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [
            {
                "generated_text": list(messages)
                + [{"role": "assistant", "content": self.answer}]
            }
        ]


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = health_chatbot.build_messages("hello")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], health_chatbot.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "hello"})

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        messages = health_chatbot.build_messages("c", history)
        self.assertEqual(
            [m["role"] for m in messages], ["system", "user", "assistant", "user"]
        )

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            health_chatbot.build_messages("   ")

    def test_bad_history_turn_is_rejected(self):
        with self.assertRaises(ValueError):
            health_chatbot.build_messages("q", [{"role": "system", "content": "x"}])


class GenerateReplyTest(unittest.TestCase):
    def test_uses_generator_and_appends_disclaimer(self):
        fake = FakeGenerator()
        reply = health_chatbot.generate_reply("cough and fever", generator=fake)
        self.assertTrue(reply.startswith("Track the dates"))
        self.assertIn("not a diagnosis", reply)
        self.assertEqual(len(fake.calls), 1)

    def test_safety_prompt_is_sent_to_the_model(self):
        fake = FakeGenerator()
        health_chatbot.generate_reply("cough", generator=fake)
        sent, kwargs = fake.calls[0]
        self.assertEqual(sent[0]["content"], health_chatbot.SYSTEM_PROMPT)
        self.assertIn("Do not diagnose", sent[0]["content"])
        self.assertFalse(kwargs["do_sample"])

    def test_disclaimer_is_not_duplicated(self):
        fake = FakeGenerator(answer="Already safe." + health_chatbot.DISCLAIMER)
        reply = health_chatbot.generate_reply("q", generator=fake)
        self.assertEqual(reply.count("not a diagnosis"), 1)

    def test_empty_model_answer_is_rejected(self):
        fake = FakeGenerator(answer="   ")
        with self.assertRaises(ValueError):
            health_chatbot.generate_reply("q", generator=fake)


class PromptFallbackTest(unittest.TestCase):
    def test_fallback_prompt_contains_roles(self):
        prompt = health_chatbot.format_prompt(health_chatbot.build_messages("q"))
        self.assertIn("system:", prompt)
        self.assertTrue(prompt.endswith("assistant:"))


if __name__ == "__main__":
    unittest.main()
