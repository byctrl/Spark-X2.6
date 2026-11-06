import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("halo")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "halo"})

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "tool", "content": "ignored"},
        ]
        messages = hc.build_messages("c", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "c")

    def test_history_is_not_mutated(self):
        history = [{"role": "user", "content": "a"}]
        hc.build_messages("b", history)
        self.assertEqual(history, [{"role": "user", "content": "a"}])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class PayloadTest(unittest.TestCase):
    def test_payload_targets_spark_model(self):
        payload = hc.build_payload(hc.build_messages("halo"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])


class ParseResponseTest(unittest.TestCase):
    def test_extracts_and_strips_content(self):
        data = {"choices": [{"message": {"content": "  jawaban  "}}]}
        self.assertEqual(hc.parse_response(data), "jawaban")

    def test_missing_choices_raise(self):
        with self.assertRaises(ValueError):
            hc.parse_response({"choices": []})

    def test_empty_content_raises(self):
        with self.assertRaises(ValueError):
            hc.parse_response({"choices": [{"message": {"content": "  "}}]})


class SafetyPromptTest(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_points_to_clinician(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("diagnosis", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("prescription", prompt)

    def test_demo_question_is_present(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


if __name__ == "__main__":
    unittest.main()
