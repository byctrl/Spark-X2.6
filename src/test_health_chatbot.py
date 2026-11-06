"""Small unit tests for src/health_chatbot.py (no model server required)."""

import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("Apa itu demam?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Apa itu demam?"})

    def test_history_is_copied_and_filtered(self):
        history = [
            {"role": "user", "content": "Halo"},
            {"role": "assistant", "content": "Halo, ada yang bisa saya bantu?"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("Saya batuk", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(len(history), 4)

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class SafetyTest(unittest.TestCase):
    def test_red_flag_detected_case_insensitive(self):
        self.assertTrue(hc.needs_urgent_care("Saya nyeri dada sejak pagi"))
        self.assertTrue(hc.needs_urgent_care("I have CHEST PAIN"))

    def test_ordinary_question_not_flagged(self):
        self.assertFalse(hc.needs_urgent_care(hc.DEMO_QUESTION))
        self.assertFalse(hc.needs_urgent_care(None))

    def test_urgent_question_skips_model(self):
        calls = []

        def fake_client(question, history=None):
            calls.append(question)
            return "model answer"

        answer = hc.respond("sulit bernapas", client=fake_client)
        self.assertEqual(answer, hc.URGENT_REPLY)
        self.assertEqual(calls, [])


class RespondTest(unittest.TestCase):
    def test_answer_gets_disclaimer(self):
        answer = hc.respond("Apa itu demam?", client=lambda q, h=None: "  Demam adalah ...  ")
        self.assertTrue(answer.startswith("Demam adalah"))
        self.assertTrue(answer.endswith(hc.DISCLAIMER))

    def test_empty_model_output_has_fallback(self):
        answer = hc.respond("Apa itu demam?", client=lambda q, h=None: "   ")
        self.assertIn("empty answer", answer)
        self.assertTrue(answer.endswith(hc.DISCLAIMER))


class ConfigTest(unittest.TestCase):
    def test_model_config(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertIsInstance(hc.MAX_NEW_TOKENS, int)
        self.assertGreater(hc.MAX_NEW_TOKENS, 0)
        self.assertIn("not medical advice", hc.DISCLAIMER)
        self.assertIn("do NOT diagnose", hc.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
