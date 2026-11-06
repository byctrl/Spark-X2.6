"""Offline tests for the Spark-X2.6-4B health-information chatbot.

No model weights are downloaded: :func:`generate` is exercised with an
injected fake generator.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc  # noqa: E402


class FakeGenerator:
    def __init__(self, reply="Rest, drink water and see a clinician if it worsens.", chat=True):
        self.reply = reply
        self.chat = chat
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.chat:
            return [
                {
                    "generated_text": list(messages)
                    + [{"role": "assistant", "content": self.reply}]
                }
            ]
        return [{"generated_text": self.reply}]


class ConstantsTest(unittest.TestCase):
    def test_model_id_points_to_spark(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_generation_budget(self):
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_safety_rules(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("emergency", prompt)

    def test_demo_question_is_present(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())
        self.assertIn("fever", hc.DEMO_QUESTION.lower())

    def test_disclaimer_text(self):
        self.assertIn("not medical advice", hc.DISCLAIMER.lower())


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("What is dehydration?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is dehydration?"})

    def test_question_is_stripped(self):
        messages = hc.build_messages("  Is a fever dangerous?  ")
        self.assertEqual(messages[-1]["content"], "Is a fever dangerous?")

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = hc.build_messages("Three days.", history=history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "I have a cough.")

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_invalid_history_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("Hi", history=[{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("Hi", history=[{"role": "user", "content": 1}])
        with self.assertRaises(ValueError):
            hc.build_messages("Hi", history=["not a dict"])


class GenerateTest(unittest.TestCase):
    def test_chat_style_output_is_extracted(self):
        fake = FakeGenerator(reply="Please see a clinician.")
        reply = hc.generate(hc.DEMO_QUESTION, generator=fake)
        self.assertEqual(reply, "Please see a clinician.")
        messages, kwargs = fake.calls[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(kwargs["max_new_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(kwargs["do_sample"])

    def test_plain_text_output_is_extracted(self):
        fake = FakeGenerator(reply="  Drink water.  ", chat=False)
        self.assertEqual(hc.generate("Thirsty?", generator=fake), "Drink water.")

    def test_custom_token_budget_is_forwarded(self):
        fake = FakeGenerator()
        hc.generate("Hi", generator=fake, max_new_tokens=64)
        self.assertEqual(fake.calls[0][1]["max_new_tokens"], 64)

    def test_history_is_forwarded(self):
        fake = FakeGenerator()
        history = [{"role": "user", "content": "I have a cough."}]
        hc.generate("Since yesterday.", history=history, generator=fake)
        roles = [m["role"] for m in fake.calls[0][0]]
        self.assertEqual(roles, ["system", "user", "user"])

    def test_empty_reply_raises(self):
        fake = FakeGenerator(reply="   ", chat=False)
        with self.assertRaises(RuntimeError):
            hc.generate("Hi", generator=fake)

    def test_invalid_token_budget_raises(self):
        for bad in (0, -5, None, "many"):
            with self.assertRaises(ValueError):
                hc.generate("Hi", generator=FakeGenerator(), max_new_tokens=bad)


if __name__ == "__main__":
    unittest.main()
