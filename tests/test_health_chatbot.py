"""Tests for the Spark-X2.6-4B health-information chatbot demo."""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import health_chatbot as hc  # noqa: E402


class PromptTests(unittest.TestCase):
    def test_system_prompt_forbids_diagnosis_and_prescribing(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("do not prescribe", prompt)
        self.assertIn("qualified clinician", prompt)

    def test_model_id_points_at_spark_x2_6_4b(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_demo_question_is_not_flagged_as_emergency(self):
        self.assertFalse(hc.is_emergency(hc.DEMO_QUESTION))


class MessageTests(unittest.TestCase):
    def test_build_messages_starts_with_system_and_ends_with_question(self):
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], hc.DEMO_QUESTION)

    def test_build_messages_adds_emergency_note(self):
        messages = hc.build_messages("I have chest pain", emergency=True)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["content"], hc.EMERGENCY_NOTE)

    def test_build_messages_keeps_history(self):
        history = [
            {"role": "user", "content": "I have a cough"},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = hc.build_messages("About a week", history=history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"], "I have a cough")

    def test_build_messages_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_build_messages_rejects_bad_history_entries(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hi", history=[{"role": "system", "content": "x"}])


class PayloadTests(unittest.TestCase):
    def test_payload_targets_spark_model(self):
        payload = hc.build_payload(hc.build_messages(hc.DEMO_QUESTION))
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])

    def test_extract_answer_reads_choice_content(self):
        response = {"choices": [{"message": {"content": "  hello  "}}]}
        self.assertEqual(hc.extract_answer(response), "hello")

    def test_extract_answer_rejects_empty_response(self):
        with self.assertRaises(ValueError):
            hc.extract_answer({"choices": []})

    def test_disclaimer_is_appended(self):
        text = hc.with_disclaimer("Some info.")
        self.assertTrue(text.startswith("Some info."))
        self.assertIn(hc.DISCLAIMER, text)


class AnswerQuestionTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.original = hc.request_completion

    def tearDown(self):
        hc.request_completion = self.original

    def _fake(self, messages, base_url=None, max_new_tokens=hc.MAX_NEW_TOKENS):
        self.calls.append({"messages": messages, "base_url": base_url})
        return {"choices": [{"message": {"content": "General information only."}}]}

    def test_answer_question_adds_disclaimer(self):
        hc.request_completion = self._fake
        result = hc.answer_question(hc.DEMO_QUESTION)
        self.assertFalse(result["emergency"])
        self.assertEqual(result["model"], hc.MODEL_ID)
        self.assertIn(hc.DISCLAIMER, result["answer"])
        self.assertEqual(len(self.calls), 1)

    def test_answer_question_flags_emergency(self):
        hc.request_completion = self._fake
        result = hc.answer_question("My father is unconscious and cannot breathe")
        self.assertTrue(result["emergency"])
        system_notes = [
            m["content"] for m in self.calls[0]["messages"] if m["role"] == "system"
        ]
        self.assertIn(hc.EMERGENCY_NOTE, system_notes)


if __name__ == "__main__":
    unittest.main()
