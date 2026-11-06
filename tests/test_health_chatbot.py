"""Standard-library tests for src/health_chatbot.py (no model, no network)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc  # noqa: E402


class PromptTests(unittest.TestCase):
    def test_model_id_targets_spark_x2_6_4b(self) -> None:
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_system_prompt_forbids_diagnosis_and_prescription(self) -> None:
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("never prescribe", prompt)
        self.assertIn("seek care", prompt)

    def test_build_messages_has_system_then_user(self) -> None:
        messages = hc.build_messages("  Apa itu demam berdarah?  ")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "Apa itu demam berdarah?")

    def test_build_messages_rejects_empty_and_oversized_input(self) -> None:
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            hc.build_messages("a" * (hc.MAX_QUESTION_CHARS + 1))

    def test_build_payload_uses_model_and_token_budget(self) -> None:
        payload = hc.build_payload("Saya batuk.")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertGreater(payload["max_tokens"], 0)
        self.assertEqual(payload["messages"][-1]["content"], "Saya batuk.")


class ReplyParsingTests(unittest.TestCase):
    def test_extract_reply_reads_message_content(self) -> None:
        payload = {"choices": [{"message": {"content": "  Minum air cukup.  "}}]}
        self.assertEqual(hc.extract_reply(payload), "Minum air cukup.")

    def test_extract_reply_reads_legacy_text_field(self) -> None:
        payload = {"choices": [{"text": "Hubungi klinik."}]}
        self.assertEqual(hc.extract_reply(payload), "Hubungi klinik.")

    def test_extract_reply_rejects_unusable_payloads(self) -> None:
        bad_payloads = [
            None,
            "not-an-object",
            {},
            {"choices": []},
            {"choices": "nope"},
            {"choices": [{"message": {"content": "   "}}]},
            {"choices": [{"message": {"content": None}}]},
        ]
        for payload in bad_payloads:
            with self.assertRaises(hc.ChatbotError):
                hc.extract_reply(payload)


class EndpointTests(unittest.TestCase):
    def test_ask_reports_unreachable_endpoint_as_chatbot_error(self) -> None:
        with self.assertRaises(hc.ChatbotError):
            hc.ask("Saya demam.", base_url="http://127.0.0.1:1/v1", timeout=2)

    def test_ask_validates_question_before_network(self) -> None:
        with self.assertRaises(ValueError):
            hc.ask("   ", base_url="http://127.0.0.1:1/v1", timeout=2)


if __name__ == "__main__":
    unittest.main()
