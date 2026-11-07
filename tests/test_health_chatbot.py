"""Offline tests for src/health_chatbot.py (no model server required)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_and_question_included(self):
        messages = hc.build_messages("  Apa itu demam?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Apa itu demam?"})

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")
        with self.assertRaises(ValueError):
            hc.build_messages(None)

    def test_history_is_filtered_and_kept_in_order(self):
        history = [
            {"role": "user", "content": "Halo"},
            {"role": "assistant", "content": "Halo, ada yang bisa saya bantu?"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("Pertanyaan baru", history)
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "Pertanyaan baru")

    def test_system_prompt_forbids_diagnosis_and_prescription(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never give a diagnosis", prompt)
        self.assertIn("prescription", prompt)
        self.assertIn("emergency", prompt)


class PayloadTest(unittest.TestCase):
    def test_payload_uses_model_id_and_token_limit(self):
        payload = hc.build_payload(hc.build_messages("q"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])


class ExtractReplyTest(unittest.TestCase):
    def test_reads_content(self):
        data = {"choices": [{"message": {"content": "  jawaban  "}}]}
        self.assertEqual(hc.extract_reply(data), "jawaban")

    def test_empty_content_raises(self):
        with self.assertRaises(RuntimeError):
            hc.extract_reply({"choices": [{"message": {"content": "  "}}]})

    def test_bad_shape_raises(self):
        with self.assertRaises(RuntimeError):
            hc.extract_reply({"unexpected": True})


class UnreachableServerTest(unittest.TestCase):
    def test_unreachable_server_gives_actionable_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            hc.chat("halo", base_url="http://127.0.0.1:1/v1", timeout=2)
        self.assertIn("Spark-X2.6-4B", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
