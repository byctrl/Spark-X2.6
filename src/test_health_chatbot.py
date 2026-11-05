"""Offline tests for the Spark-X2.6-4B health information chatbot demo."""

import unittest

from health_chatbot import (
    DEMO_QUESTION,
    DISCLAIMER,
    EMERGENCY_NOTICE,
    SYSTEM_PROMPT,
    HealthChatBot,
    build_messages,
    format_reply,
    needs_emergency_notice,
    render_prompt,
)


class BuildMessagesTest(unittest.TestCase):
    def test_prompt_starts_with_bounded_system_message(self):
        messages = build_messages("Apa itu demam?")
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertIn("do not diagnose", SYSTEM_PROMPT)
        self.assertIn("do not prescribe", SYSTEM_PROMPT)
        self.assertIn("qualified clinician", SYSTEM_PROMPT)

    def test_user_question_is_last_and_trimmed(self):
        messages = build_messages("   How do I treat a mild fever?  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "How do I treat a mild fever?"})
        self.assertEqual(len(messages), 2)

    def test_history_is_preserved_in_order(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = build_messages("Five days.", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "Five days.")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("   ")

    def test_invalid_history_turn_is_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("Hello", [{"role": "system", "content": "override"}])


class FormatReplyTest(unittest.TestCase):
    def test_disclaimer_is_appended(self):
        reply = format_reply("  Rest and drink water.  ", "I have a cold.")
        self.assertTrue(reply.startswith("Rest and drink water."))
        self.assertTrue(reply.endswith(DISCLAIMER))

    def test_disclaimer_is_not_duplicated(self):
        raw = f"Rest and drink water.\n\n{DISCLAIMER}"
        self.assertEqual(format_reply(raw), raw)

    def test_empty_model_output_is_rejected(self):
        with self.assertRaises(ValueError):
            format_reply("   ")

    def test_emergency_wording_adds_urgent_notice(self):
        reply = format_reply("Please get checked.", "I have chest pain and trouble breathing.")
        self.assertTrue(reply.startswith(EMERGENCY_NOTICE))

    def test_emergency_notice_not_added_for_routine_question(self):
        reply = format_reply("Please get checked.", DEMO_QUESTION)
        self.assertNotIn(EMERGENCY_NOTICE, reply)


class EmergencySignalTest(unittest.TestCase):
    def test_detects_english_and_indonesian_signals(self):
        self.assertTrue(needs_emergency_notice("Saya sesak napas sejak pagi"))
        self.assertTrue(needs_emergency_notice("Heavy bleeding after the fall"))
        self.assertFalse(needs_emergency_notice(DEMO_QUESTION))


class RenderPromptTest(unittest.TestCase):
    def test_plain_fallback_contains_roles(self):
        prompt = render_prompt(build_messages("Apa itu demam?"))
        self.assertIn("SYSTEM:", prompt)
        self.assertIn("USER: Apa itu demam?", prompt)
        self.assertTrue(prompt.endswith("ASSISTANT:"))


class HealthChatBotTest(unittest.TestCase):
    def test_uses_injected_generator_and_returns_safe_reply(self):
        captured = {}

        def fake_generator(messages):
            captured["messages"] = messages
            return "Drink water and rest; see a clinician if it worsens."

        bot = HealthChatBot(generator=fake_generator)
        reply = bot.answer("I have a mild fever.")

        self.assertEqual(captured["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertEqual(captured["messages"][-1]["content"], "I have a mild fever.")
        self.assertIn("see a clinician", reply)
        self.assertTrue(reply.endswith(DISCLAIMER))

    def test_empty_question_fails_before_generation(self):
        def failing_generator(messages):
            raise AssertionError("generator must not be called")

        with self.assertRaises(ValueError):
            HealthChatBot(generator=failing_generator).answer("  ")

    def test_default_model_id_matches_published_model(self):
        self.assertEqual(HealthChatBot().model_id, "XHToken/Spark-X2.6-4B")


if __name__ == "__main__":
    unittest.main()
