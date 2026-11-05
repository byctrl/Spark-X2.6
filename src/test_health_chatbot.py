"""Offline tests for the Spark-X2.6-4B health-information chatbot demo."""

import unittest

import health_chatbot as hc


class FakeTransport:
    def __init__(self, answer="General information about coughs."):
        self.answer = answer
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append({"url": url, "payload": payload, "timeout": timeout})
        return {"choices": [{"message": {"role": "assistant", "content": self.answer}}]}


class MessageTests(unittest.TestCase):
    def test_system_prompt_forbids_diagnosis_and_prescription(self):
        self.assertIn("Do not diagnose", hc.SYSTEM_PROMPT)
        self.assertIn("Do not prescribe", hc.SYSTEM_PROMPT)
        self.assertIn("emergency", hc.SYSTEM_PROMPT.lower())

    def test_build_messages_places_system_prompt_first(self):
        messages = hc.build_messages(hc.DEMO_QUESTION)
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": hc.DEMO_QUESTION.strip()})

    def test_build_messages_keeps_valid_history_and_drops_invalid_turns(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("question", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_build_messages_rejects_empty_question(self):
        for question in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(question)

    def test_payload_targets_spark_model_and_token_budget(self):
        payload = hc.build_payload("What is a fever?")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)


class ResponseTests(unittest.TestCase):
    def test_extract_answer_returns_stripped_text(self):
        response = {"choices": [{"message": {"content": "  rest and fluids  "}}]}
        self.assertEqual(hc.extract_answer(response), "rest and fluids")

    def test_extract_answer_rejects_empty_and_malformed(self):
        for response in (
            {"choices": [{"message": {"content": "   "}}]},
            {"choices": []},
            {},
            None,
        ):
            with self.assertRaises(hc.ChatError):
                hc.extract_answer(response)

    def test_ask_calls_chat_completions_and_adds_disclaimer(self):
        transport = FakeTransport("Rest and drink fluids; see a clinician if it worsens.")
        answer = hc.ask("What helps a cough?", transport=transport, base_url="http://localhost:30000/v1/")
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0]["url"], "http://localhost:30000/v1/chat/completions")
        self.assertIn("Rest and drink fluids", answer)
        self.assertIn(hc.DISCLAIMER, answer)
        self.assertNotIn(hc.EMERGENCY_NOTICE, answer)

    def test_ask_prepends_emergency_notice_for_red_flags(self):
        transport = FakeTransport("Please seek care promptly.")
        answer = hc.ask("I have chest pain and cannot breathe", transport=transport)
        self.assertTrue(answer.startswith(hc.EMERGENCY_NOTICE))
        self.assertIn(hc.DISCLAIMER, answer)

    def test_needs_emergency_notice_is_case_insensitive(self):
        self.assertTrue(hc.needs_emergency_notice("Sudden CHEST PAIN"))
        self.assertFalse(hc.needs_emergency_notice("Mild cough for four days"))


if __name__ == "__main__":
    unittest.main()
