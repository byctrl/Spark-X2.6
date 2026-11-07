"""Offline tests for src/health_chatbot.py (no model download required)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_and_question(self):
        messages = hc.build_messages("What is a normal blood pressure?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("patient-education assistant", messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a normal blood pressure?"})
        self.assertEqual(len(messages), 2)

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_is_kept_in_order_and_filtered(self):
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": ""},
        ]
        messages = hc.build_messages("second", history)
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "second")


class FinalizeAnswerTest(unittest.TestCase):
    def test_disclaimer_appended_once(self):
        answer = hc.finalize_answer("Rest and drink water.", "I feel tired.")
        self.assertTrue(answer.startswith("Rest and drink water."))
        self.assertIn(hc.DISCLAIMER, answer)
        self.assertEqual(answer.count(hc.DISCLAIMER), 1)

    def test_empty_answer_is_replaced(self):
        answer = hc.finalize_answer("   ", "Hello")
        self.assertIn("could not produce an answer", answer)
        self.assertIn(hc.DISCLAIMER, answer)

    def test_emergency_wording_adds_notice(self):
        answer = hc.finalize_answer("Try resting.", "I have chest pain right now")
        self.assertTrue(answer.startswith(hc.EMERGENCY_NOTICE))

    def test_no_duplicate_emergency_notice(self):
        text = "Go to the emergency department now."
        answer = hc.finalize_answer(text, "I have chest pain")
        self.assertFalse(answer.startswith(hc.EMERGENCY_NOTICE))

    def test_emergency_screen(self):
        self.assertTrue(hc.is_emergency_wording("My baby will not feed"))
        self.assertFalse(hc.is_emergency_wording("Where can I read about vaccines?"))


class ChatbotTest(unittest.TestCase):
    def test_ask_uses_injected_backend_and_keeps_history(self):
        seen = []

        def fake_complete(messages):
            seen.append(messages)
            return "Drink water and rest."

        bot = hc.HealthChatbot(complete=fake_complete)
        answer = bot.ask("I have a mild headache.")

        self.assertIn("Drink water and rest.", answer)
        self.assertIn(hc.DISCLAIMER, answer)
        self.assertEqual(seen[0][-1]["content"], "I have a mild headache.")
        self.assertEqual([t["role"] for t in bot.history], ["user", "assistant"])

        bot.ask("And now?")
        roles = [m["role"] for m in seen[1]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])


class ConstantsTest(unittest.TestCase):
    def test_model_and_budget(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)
        self.assertTrue(hc.DEMO_QUESTION)

    def test_system_prompt_forbids_prescribing(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("must not diagnose", prompt)
        self.assertIn("emergency", prompt)


if __name__ == "__main__":
    unittest.main()
