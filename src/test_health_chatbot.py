"""Unit tests for the Spark-X2.6-4B health information chatbot.

Run from the repository root:

    python -m unittest discover -s src -p "test_*.py" -t src
"""

import unittest

import health_chatbot as hc


class FakeClient:
    def __init__(self, reply="Ask your clinician about this."):
        self.reply = reply
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.reply


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_is_first(self):
        messages = hc.build_messages("How much water should I drink?")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})

    def test_question_is_last_user_turn(self):
        messages = hc.build_messages("How much water should I drink?")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("water", messages[-1]["content"])

    def test_history_is_kept_between_system_and_question(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi, how can I help?"},
        ]
        messages = hc.build_messages("and now?", history=history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1:3], history)

    def test_question_is_stripped(self):
        messages = hc.build_messages("  sore throat  ")
        self.assertEqual(messages[-1]["content"], "sore throat")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class EmergencyTest(unittest.TestCase):
    def test_red_flag_is_detected(self):
        self.assertTrue(hc.is_emergency("I have crushing chest pain right now"))

    def test_ordinary_question_is_not_red_flag(self):
        self.assertFalse(hc.is_emergency(hc.DEMO_QUESTION))

    def test_emergency_skips_the_model(self):
        client = FakeClient()
        reply = hc.answer("I cannot breathe and I fainted.", client=client)
        self.assertEqual(client.calls, [])
        self.assertIn("emergency", reply.lower())


class AnswerTest(unittest.TestCase):
    def test_model_is_called_for_ordinary_question(self):
        client = FakeClient("Rest and fluids may help. " + hc.DISCLAIMER)
        reply = hc.answer("I have a mild cough.", client=client)
        self.assertEqual(len(client.calls), 1)
        self.assertIn("general information", reply)

    def test_disclaimer_is_appended_when_missing(self):
        client = FakeClient("Drink water.")
        reply = hc.answer("I feel thirsty.", client=client)
        self.assertTrue(reply.endswith(hc.DISCLAIMER))

    def test_disclaimer_is_not_duplicated(self):
        client = FakeClient("Drink water. " + hc.DISCLAIMER)
        reply = hc.answer("I feel thirsty.", client=client)
        self.assertEqual(reply.lower().count(hc.DISCLAIMER.lower()), 1)

    def test_empty_model_reply_gets_fallback(self):
        client = FakeClient("   ")
        reply = hc.answer("I feel thirsty.", client=client)
        self.assertIn("could not produce an answer", reply)
        self.assertIn(hc.DISCLAIMER, reply)

    def test_model_id_and_token_budget(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)


if __name__ == "__main__":
    unittest.main()
