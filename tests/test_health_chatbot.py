"""Tests for the Spark Health Guide demo (no network, no GPU needed)."""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first_and_question_last(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_turns_are_kept_in_order(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "ignored"},
        ]
        messages = hc.build_messages("next", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_system_prompt_blocks_diagnosis_and_prescribing(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("must not: diagnose", prompt)
        self.assertIn("prescribe", prompt)
        self.assertIn("emergency", prompt)


class PayloadTest(unittest.TestCase):
    def test_payload_is_bounded_and_uses_the_target_model(self):
        payload = hc.build_payload("How much water should I drink?")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertLessEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["messages"][-1]["role"], "user")


class HealthChatbotTest(unittest.TestCase):
    def test_ask_returns_answer_and_disclaimer(self):
        seen = {}

        def fake_transport(payload):
            seen["payload"] = payload
            return "  A fever is a common sign of infection.  \nNext step: call a local clinic.  "

        result = hc.HealthChatbot(fake_transport).ask("What is a fever?")
        self.assertTrue(result["answer"].startswith("A fever is a common sign"))
        self.assertFalse(result["emergency"])
        self.assertEqual(result["disclaimer"], hc.DISCLAIMER)
        self.assertEqual(seen["payload"]["model"], hc.MODEL_ID)

    def test_empty_model_response_raises(self):
        with self.assertRaises(RuntimeError):
            hc.HealthChatbot(lambda payload: "   ").ask("What is a fever?")

    def test_non_callable_transport_is_rejected(self):
        with self.assertRaises(TypeError):
            hc.HealthChatbot("not a transport")

    def test_emergency_question_gets_a_hardcoded_banner(self):
        def fake_transport(payload):
            return "Please seek care soon.\nNext step: see a clinician."

        result = hc.HealthChatbot(fake_transport).ask("I have chest pain right now")
        self.assertTrue(result["emergency"])
        self.assertTrue(result["answer"].startswith("EMERGENCY:"))
        self.assertIn("chest pain right now", result["question"])

    def test_model_leading_emergency_is_not_duplicated(self):
        def fake_transport(payload):
            return "EMERGENCY: call 112 now."

        result = hc.HealthChatbot(fake_transport).ask("my father is unconscious")
        self.assertEqual(result["answer"].count("EMERGENCY:"), 1)


class EmergencyDetectionTest(unittest.TestCase):
    def test_detects_red_flags(self):
        self.assertTrue(hc.is_emergency("I cannot breathe well"))
        self.assertTrue(hc.is_emergency("HEAVY BLEEDING"))

    def test_ignores_ordinary_questions(self):
        self.assertFalse(hc.is_emergency("How do I book a dental appointment?"))
        self.assertFalse(hc.is_emergency(None))


if __name__ == "__main__":
    unittest.main()
