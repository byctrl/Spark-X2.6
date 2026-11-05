"""Tests for the minimal Spark-X2.6-4B health-information chatbot."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first_and_question_last(self):
        messages = hc.build_messages("Apa itu demam?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("clinician", messages[0]["content"].lower())
        self.assertEqual(
            messages[-1], {"role": "user", "content": "Apa itu demam?"}
        )

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_is_kept_and_filtered(self):
        history = [
            {"role": "user", "content": "halo"},
            {"role": "assistant", "content": "hai"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": ""},
        ]
        roles = [m["role"] for m in hc.build_messages("Q", history)]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])


class PayloadTest(unittest.TestCase):
    def test_model_and_token_budget(self):
        payload = hc.build_payload(hc.build_messages("Q"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)


class ParseResponseTest(unittest.TestCase):
    def test_extracts_trimmed_text(self):
        data = {"choices": [{"message": {"content": "  jawaban  "}}]}
        self.assertEqual(hc.parse_response(data), "jawaban")

    def test_rejects_missing_choices(self):
        with self.assertRaises(ValueError):
            hc.parse_response({"choices": []})

    def test_rejects_blank_content(self):
        with self.assertRaises(ValueError):
            hc.parse_response({"choices": [{"message": {"content": "   "}}]})


class AskTest(unittest.TestCase):
    def test_uses_transport_and_returns_answer(self):
        captured = {}

        def fake_transport(url, payload):
            captured["url"] = url
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "Jawaban"}}]}

        answer = hc.ask("Q", transport=fake_transport)
        self.assertEqual(answer, "Jawaban")
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["payload"]["model"], hc.MODEL_ID)


if __name__ == "__main__":
    unittest.main()
