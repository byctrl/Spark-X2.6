"""Tests for the minimal Spark-X2.6-4B health chatbot demo.

Run from the repository root::

    python -m unittest discover -s src -p "test_*.py"

No model server or network access is required.
"""

import unittest

import health_chatbot as hc


class BuildMessagesTests(unittest.TestCase):
    def test_system_prompt_comes_first_and_question_last(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = hc.build_messages("Two weeks.", history=history)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "I have a cough.")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")
        with self.assertRaises(ValueError):
            hc.build_messages(None)

    def test_invalid_history_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hi", history=["not-a-dict"])
        with self.assertRaises(ValueError):
            hc.build_messages("hi", history=[{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("hi", history=[{"role": "user", "content": "  "}])

    def test_system_prompt_forbids_diagnosis_and_prescribing(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("do not prescribe", prompt)
        self.assertIn("emergency", prompt)


class UrgentMarkerTests(unittest.TestCase):
    def test_urgent_markers_are_detected(self):
        self.assertTrue(hc.is_urgent("I have chest pain right now"))
        self.assertTrue(hc.is_urgent("She is unconscious"))
        self.assertTrue(hc.is_urgent("突然胸痛怎么办"))

    def test_ordinary_question_is_not_urgent(self):
        self.assertFalse(hc.is_urgent("What is a mild cough?"))
        self.assertFalse(hc.is_urgent(""))
        self.assertFalse(hc.is_urgent(None))


class ExtractContentTests(unittest.TestCase):
    def test_reads_assistant_content(self):
        response = {"choices": [{"message": {"content": "  hello  "}}]}
        self.assertEqual(hc.extract_content(response), "hello")

    def test_rejects_malformed_responses(self):
        for response in (
            None,
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {"content": "   "}}]},
        ):
            with self.assertRaises(ValueError):
                hc.extract_content(response)


class ChatTests(unittest.TestCase):
    def _transport(self, captured):
        def sender(url, payload, timeout):
            captured.append((url, payload, timeout))
            return {"choices": [{"message": {"content": "General guidance."}}]}

        return sender

    def test_payload_and_endpoint(self):
        captured = []
        reply = hc.chat(
            "What is a fever?",
            base_url="http://127.0.0.1:30000/v1/",
            transport=self._transport(captured),
        )
        url, payload, _timeout = captured[0]
        self.assertEqual(url, "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(payload["messages"][-1]["content"], "What is a fever?")
        self.assertEqual(reply.answer, "General guidance.")
        self.assertFalse(reply.urgent)
        self.assertEqual(reply.model, hc.MODEL_ID)

    def test_urgent_question_gets_escalation_notice(self):
        captured = []
        reply = hc.chat(
            "I have chest pain and cannot breathe",
            transport=self._transport(captured),
        )
        self.assertTrue(reply.urgent)
        self.assertTrue(reply.answer.startswith(hc.URGENT_NOTICE))
        self.assertIn("General guidance.", reply.answer)

    def test_model_override(self):
        captured = []
        reply = hc.chat("hi", model="local-q4", transport=self._transport(captured))
        self.assertEqual(captured[0][1]["model"], "local-q4")
        self.assertEqual(reply.model, "local-q4")

    def test_empty_model_output_raises(self):
        def sender(url, payload, timeout):
            return {"choices": [{"message": {"content": ""}}]}

        with self.assertRaises(ValueError):
            hc.chat("hi", transport=sender)


if __name__ == "__main__":
    unittest.main()
