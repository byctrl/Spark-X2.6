"""Offline tests for src/health_chatbot.py (no model download, no network)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first_and_question_last(self):
        messages = hc.build_messages("  demam tiga hari  ")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "demam tiga hari"})

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "Apa itu demam?"},
            {"role": "assistant", "content": "Demam adalah suhu tubuh di atas normal."},
        ]
        messages = hc.build_messages("Berapa lama?", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[3]["content"], "Berapa lama?")

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_invalid_history_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("halo", [{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("halo", [{"role": "user", "content": "  "}])


class PayloadTest(unittest.TestCase):
    def test_payload_uses_model_and_token_budget(self):
        payload = hc.build_payload(hc.build_messages("halo"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])

    def test_system_prompt_blocks_diagnosis_and_points_to_clinician(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("not diagnose", prompt)
        self.assertIn("prescribe", prompt)
        self.assertIn("clinician", prompt)


class GenerateTest(unittest.TestCase):
    def test_generate_with_fake_transport(self):
        seen = {}

        def fake_transport(payload, base_url, api_key, timeout):
            seen["payload"] = payload
            seen["base_url"] = base_url
            seen["api_key"] = api_key
            seen["timeout"] = timeout
            return {"choices": [{"message": {"role": "assistant",
                                             "content": "  Segera temui dokter.  "}}]}

        answer = hc.generate(
            "batuk dan demam",
            base_url="http://example.test/v1",
            api_key="test-key",
            timeout=5,
            transport=fake_transport,
        )

        self.assertEqual(answer, "Segera temui dokter.")
        self.assertEqual(seen["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(seen["payload"]["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(seen["payload"]["messages"][-1]["content"], "batuk dan demam")
        self.assertEqual(seen["base_url"], "http://example.test/v1")
        self.assertEqual(seen["api_key"], "test-key")
        self.assertEqual(seen["timeout"], 5)

    def test_empty_answer_is_rejected(self):
        with self.assertRaises(RuntimeError):
            hc.parse_answer({"choices": [{"message": {"content": "   "}}]})

    def test_malformed_answer_is_rejected(self):
        with self.assertRaises(RuntimeError):
            hc.parse_answer({})

    def test_format_reply_appends_disclaimer(self):
        reply = hc.format_reply("Minum air yang cukup.")
        self.assertTrue(reply.startswith("Minum air yang cukup."))
        self.assertIn(hc.DISCLAIMER, reply)


if __name__ == "__main__":
    unittest.main()
