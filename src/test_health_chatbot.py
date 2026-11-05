"""Small tests for the health chatbot demo.

These tests never load the Spark-X2.6-4B weights; they only cover prompt
construction, input validation and the safety disclaimer.
"""

import unittest

try:  # works both with `-s src` discovery and `python -m unittest src...`
    from health_chatbot import (
        DEMO_QUESTION,
        DISCLAIMER,
        MODEL_ID,
        SYSTEM_PROMPT,
        build_messages,
        format_reply,
    )
    import health_chatbot
    _generated = health_chatbot.generate
    _main = health_chatbot.main
    _SAME_MODULE = True
except ImportError:  # pragma: no cover - fallback for package-style discovery
    from src.health_chatbot import (  # type: ignore
        DEMO_QUESTION,
        DISCLAIMER,
        MODEL_ID,
        SYSTEM_PROMPT,
        build_messages,
        format_reply,
    )
    import src.health_chatbot as health_chatbot  # type: ignore

    _generated = health_chatbot.generate
    _main = health_chatbot.main
    _SAME_MODULE = False


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_is_first_and_question_last(self):
        messages = build_messages("How much water should an adult drink per day?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(
            messages[-1]["content"], "How much water should an adult drink per day?"
        )

    def test_history_is_kept_in_order_and_filtered(self):
        history = [
            {"role": "user", "content": "I have a headache."},
            {"role": "assistant", "content": "Here is general information."},
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "   "},
        ]
        messages = build_messages("What are warning signs?", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "What are warning signs?")

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("   ")

    def test_system_prompt_states_safety_boundaries(self):
        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", lowered)
        self.assertIn("prescribe", lowered)
        self.assertIn("clinician", lowered)

    def test_demo_question_is_present(self):
        self.assertTrue(DEMO_QUESTION.strip())
        self.assertIn("doctor", DEMO_QUESTION.lower())


class FormatReplyTest(unittest.TestCase):
    def test_disclaimer_is_appended(self):
        reply = format_reply("  Drink water regularly.  ")
        self.assertTrue(reply.startswith("Drink water regularly."))
        self.assertTrue(reply.endswith(DISCLAIMER))

    def test_disclaimer_not_duplicated(self):
        once = format_reply("Some text.")
        twice = format_reply(once)
        self.assertEqual(once, twice)
        self.assertEqual(once.lower().count(DISCLAIMER.lower()), 1)

    def test_empty_reply_keeps_disclaimer(self):
        self.assertEqual(format_reply(""), DISCLAIMER)


class ModuleSurfaceTest(unittest.TestCase):
    def test_model_id(self):
        self.assertEqual(MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_public_entrypoints_exist(self):
        self.assertTrue(callable(_generated))
        self.assertTrue(callable(_main))


if __name__ == "__main__":
    unittest.main()
