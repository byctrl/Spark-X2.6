"""Offline tests for the Spark-X2.6-4B health-information chatbot."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("Apa itu demam?")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "Apa itu demam?")

    def test_prompt_forbids_diagnosis(self):
        prompt = hc.build_messages("halo")[0]["content"].lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("not a diagnosis", prompt)

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_non_string_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages(None)


class PayloadTest(unittest.TestCase):
    def test_payload_pins_model_and_token_budget(self):
        payload = hc.build_payload(hc.build_messages("x"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])


class ExtractAnswerTest(unittest.TestCase):
    def test_reads_and_strips_first_choice(self):
        data = {"choices": [{"message": {"content": "  halo  "}}]}
        self.assertEqual(hc.extract_answer(data), "halo")

    def test_missing_choices_raise(self):
        with self.assertRaises(ValueError):
            hc.extract_answer({})

    def test_empty_content_raises(self):
        with self.assertRaises(ValueError):
            hc.extract_answer({"choices": [{"message": {"content": "   "}}]})

    def test_non_object_response_raises(self):
        with self.assertRaises(ValueError):
            hc.extract_answer("not json")


class GenerateTest(unittest.TestCase):
    def test_generate_uses_injected_transport(self):
        seen = {}

        def fake_transport(messages):
            seen["messages"] = messages
            return {"choices": [{"message": {"content": "Jawaban uji."}}]}

        answer = hc.generate("Batuk tiga hari", transport=fake_transport)
        self.assertEqual(answer, "Jawaban uji.")
        self.assertEqual(seen["messages"][-1]["content"], "Batuk tiga hari")
        self.assertEqual(seen["messages"][0]["role"], "system")

    def test_bad_transport_response_surfaces_error(self):
        with self.assertRaises(ValueError):
            hc.generate("halo", transport=lambda messages: {"choices": []})

    def test_demo_question_matches_field_scenario(self):
        self.assertIn("demam", hc.DEMO_QUESTION.lower())
        self.assertIn("klinik", hc.DEMO_QUESTION.lower())


if __name__ == "__main__":
    unittest.main()
