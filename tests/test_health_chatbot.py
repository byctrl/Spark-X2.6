"""Tests for the minimal Spark-X2.6-4B health chatbot (no network needed)."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc  # noqa: E402


class StubBot(hc.HealthChatbot):
    def __init__(self, reply="general education answer", **kwargs):
        super().__init__(**kwargs)
        self.reply = reply
        self.payloads = []

    def _post(self, payload):
        self.payloads.append(payload)
        return {"choices": [{"message": {"role": "assistant", "content": self.reply}}]}


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_system_prompt_forbids_diagnosis_and_prescribing(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", lowered)
        self.assertIn("do not recommend prescription medicines", lowered)
        self.assertIn("clinician", lowered)

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = hc.build_messages("Two weeks.", history=history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "I have a cough.")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_invalid_history_turn_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("Hello", history=[{"role": "system", "content": "x"}])


class PayloadTest(unittest.TestCase):
    def test_payload_uses_spark_model_and_token_budget(self):
        payload = hc.build_payload(hc.build_messages("Hello"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])


class AskTest(unittest.TestCase):
    def test_ask_returns_trimmed_model_text(self):
        bot = StubBot(reply="  Drink water and rest.  ")
        self.assertEqual(
            bot.ask("I have a cough."),
            "Drink water and rest.",
        )
        self.assertEqual(len(bot.payloads), 1)
        self.assertEqual(bot.payloads[0]["model"], hc.MODEL_ID)

    def test_ask_rejects_empty_question_before_calling_model(self):
        bot = StubBot()
        with self.assertRaises(ValueError):
            bot.ask("")
        self.assertEqual(bot.payloads, [])


class ExtractAnswerTest(unittest.TestCase):
    def test_missing_choices_raises(self):
        with self.assertRaises(ValueError):
            hc.extract_answer({})

    def test_empty_content_raises(self):
        response = {"choices": [{"message": {"role": "assistant", "content": "  "}}]}
        with self.assertRaises(ValueError):
            hc.extract_answer(response)


if __name__ == "__main__":
    unittest.main()
