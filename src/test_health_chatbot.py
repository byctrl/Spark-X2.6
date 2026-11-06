"""Unit tests for the health chatbot. No model download, no network."""

import unittest

import health_chatbot as hc


class FakeGenerator:
    """Mimics the transformers chat pipeline call signature."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [{"generated_text": list(messages) + [{"role": "assistant", "content": self.answer}]}]


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_then_user(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "What is a fever?")

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_prompt_forbids_diagnosis_and_points_to_clinician(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("clinician", prompt)


class SafetyTest(unittest.TestCase):
    def test_urgent_keywords(self):
        self.assertTrue(hc.looks_urgent("I have chest pain"))
        self.assertFalse(hc.looks_urgent("I have a mild cough"))

    def test_reply_always_has_disclaimer(self):
        self.assertIn(hc.DISCLAIMER, hc.format_reply("Rest and drink water."))

    def test_reply_has_no_emergency_banner_by_default(self):
        self.assertNotIn(hc.URGENT_BANNER, hc.format_reply("Rest and drink water."))

    def test_urgent_reply_prepends_banner(self):
        reply = hc.format_reply("Seek help.", urgent=True)
        self.assertTrue(reply.startswith(hc.URGENT_BANNER))
        self.assertIn(hc.DISCLAIMER, reply)

    def test_empty_answer_gets_fallback_text(self):
        reply = hc.format_reply("")
        self.assertIn("No answer was produced", reply)


class ExtractTextTest(unittest.TestCase):
    def test_chat_style_output(self):
        output = [{"generated_text": [{"role": "assistant", "content": "Hello"}]}]
        self.assertEqual(hc.extract_text(output), "Hello")

    def test_plain_text_output(self):
        self.assertEqual(hc.extract_text([{"generated_text": "Hello"}]), "Hello")

    def test_empty_output(self):
        self.assertEqual(hc.extract_text([]), "")


class GenerateReplyTest(unittest.TestCase):
    def test_uses_bounded_generation_settings(self):
        fake = FakeGenerator("Drink water and rest.")
        reply = hc.generate_reply("What helps a cold?", generator=fake)

        messages, kwargs = fake.calls[0]
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(kwargs["max_new_tokens"], hc.MAX_NEW_TOKENS)
        self.assertIs(kwargs["do_sample"], False)
        self.assertIn("Drink water and rest.", reply)
        self.assertIn(hc.DISCLAIMER, reply)

    def test_urgent_question_gets_banner(self):
        reply = hc.generate_reply("I cannot breathe", generator=FakeGenerator("Get help."))
        self.assertIn(hc.URGENT_BANNER, reply)


if __name__ == "__main__":
    unittest.main()
