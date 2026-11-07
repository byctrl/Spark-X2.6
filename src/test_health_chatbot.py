"""Unit tests for the minimal Spark-X2.6-4B health chatbot demo.

These tests use the standard library only; the model is stubbed out so they
run offline in a few milliseconds.
"""

import unittest

import health_chatbot as hc


class PromptTest(unittest.TestCase):
    def test_messages_start_with_system_prompt(self):
        messages = hc.build_messages("Apa itu demam?")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "Apa itu demam?")

    def test_system_prompt_forbids_diagnosis_and_prescription(self):
        self.assertIn("diagnosis", hc.SYSTEM_PROMPT)
        self.assertIn("resep", hc.SYSTEM_PROMPT)
        self.assertIn("fasilitas kesehatan", hc.SYSTEM_PROMPT)

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   \n\t ")

    def test_non_string_question_is_rejected(self):
        with self.assertRaises(TypeError):
            hc.build_messages(None)

    def test_whitespace_is_collapsed(self):
        messages = hc.build_messages("  demam\n\n tiga   hari  ")
        self.assertEqual(messages[1]["content"], "demam tiga hari")

    def test_long_question_is_capped(self):
        messages = hc.build_messages("a" * (hc.MAX_INPUT_CHARS + 500))
        content = messages[1]["content"]
        self.assertLessEqual(len(content), hc.MAX_INPUT_CHARS + len(" ..."))
        self.assertTrue(content.endswith("..."))


class GenerateReplyTest(unittest.TestCase):
    def test_reply_uses_injected_generator(self):
        seen = []

        def fake_generator(messages):
            seen.append(messages)
            return "  Minum air cukup dan pantau suhu tubuh.  "

        result = hc.generate_reply("Demam tiga hari, apa yang dipantau?", fake_generator)

        self.assertEqual(result["reply"], "Minum air cukup dan pantau suhu tubuh.")
        self.assertTrue(result["used_model"])
        self.assertFalse(result["emergency"])
        self.assertEqual(result["model"], hc.MODEL_ID)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0]["content"], hc.SYSTEM_PROMPT)

    def test_empty_model_output_falls_back_to_safe_text(self):
        result = hc.generate_reply("Demam ringan", lambda messages: "   ")
        self.assertEqual(result["reply"], hc.FALLBACK_REPLY)

    def test_none_model_output_falls_back_to_safe_text(self):
        result = hc.generate_reply("Demam ringan", lambda messages: None)
        self.assertEqual(result["reply"], hc.FALLBACK_REPLY)

    def test_emergency_wording_skips_the_model(self):
        def exploding_generator(messages):
            raise AssertionError("model must not be called for emergencies")

        result = hc.generate_reply("Saya nyeri dada sejak tadi pagi", exploding_generator)

        self.assertTrue(result["emergency"])
        self.assertFalse(result["used_model"])
        self.assertEqual(result["reply"], hc.EMERGENCY_NOTICE)

    def test_emergency_detection_is_case_insensitive(self):
        self.assertTrue(hc.is_emergency("I have CHEST PAIN right now"))
        self.assertFalse(hc.is_emergency("Saya batuk kering tiga hari"))


if __name__ == "__main__":
    unittest.main()
