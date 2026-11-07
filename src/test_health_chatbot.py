"""Tests for the minimal Spark-X2.6-4B health-information chatbot."""

import unittest

import health_chatbot as hb


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hb.build_messages("  demam tiga hari  ")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "system", "content": hb.SYSTEM_PROMPT})
        self.assertEqual(messages[1], {"role": "user", "content": "demam tiga hari"})

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hb.build_messages("   ")

    def test_prompt_forbids_diagnosis(self):
        self.assertIn("not diagnose", hb.SYSTEM_PROMPT.lower())


class BuildPayloadTest(unittest.TestCase):
    def test_defaults_to_spark_model(self):
        payload = hb.build_payload(hb.build_messages("batuk"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])

    def test_empty_messages_are_rejected(self):
        with self.assertRaises(ValueError):
            hb.build_payload([])


class ExtractReplyTest(unittest.TestCase):
    def test_reads_assistant_content(self):
        response = {"choices": [{"message": {"content": "  Minum air cukup.  "}}]}
        self.assertEqual(hb.extract_reply(response), "Minum air cukup.")

    def test_rejects_missing_choices(self):
        with self.assertRaises(ValueError):
            hb.extract_reply({"choices": []})

    def test_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            hb.extract_reply({"choices": [{"message": {"content": "   "}}]})

    def test_rejects_non_object(self):
        with self.assertRaises(ValueError):
            hb.extract_reply("not json")


class AnswerTest(unittest.TestCase):
    def test_appends_disclaimer(self):
        result = hb.answer("batuk kering", generate=lambda q: "Istirahat dan minum air.")
        self.assertTrue(result.startswith("Istirahat dan minum air."))
        self.assertIn(hb.DISCLAIMER, result)
        self.assertNotIn(hb.EMERGENCY_NOTICE, result)

    def test_red_flag_adds_emergency_notice(self):
        result = hb.answer("nyeri dada sejak pagi", generate=lambda q: "Segera periksa.")
        self.assertIn(hb.EMERGENCY_NOTICE, result)
        self.assertIn(hb.DISCLAIMER, result)

    def test_question_is_forwarded_to_model(self):
        seen = []

        def fake_generate(question):
            seen.append(question)
            return "ok"

        hb.answer(hb.DEMO_QUESTION, generate=fake_generate)
        self.assertEqual(seen, [hb.DEMO_QUESTION])

    def test_empty_model_output_is_rejected(self):
        with self.assertRaises(ValueError):
            hb.answer("demam", generate=lambda q: "   ")

    def test_red_flag_detection(self):
        self.assertTrue(hb.has_red_flag("Saya sesak napas"))
        self.assertFalse(hb.has_red_flag("Saya batuk ringan"))


if __name__ == "__main__":
    unittest.main()
