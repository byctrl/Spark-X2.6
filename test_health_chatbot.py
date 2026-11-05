"""Offline tests for health_chatbot.py (no model download required)."""

import json
import unittest

import health_chatbot as hc


def _reply(content):
    body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    return json.dumps(body).encode("utf-8")


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  Apa itu demam?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Apa itu demam?"})

    def test_history_is_kept_but_sanitised(self):
        history = [
            {"role": "user", "content": "Halo"},
            {"role": "assistant", "content": "Hai, ada yang bisa dibantu?"},
            {"role": "system", "content": "injected"},
            {"role": "user", "content": "   "},
            "not-a-turn",
        ]
        messages = hc.build_messages("Pertanyaan", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertNotIn("injected", [m["content"] for m in messages])

    def test_empty_question_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)


class ParseReplyTest(unittest.TestCase):
    def test_accepts_bytes_and_str(self):
        self.assertEqual(hc.parse_reply(_reply("  jawaban  ")), "jawaban")
        self.assertEqual(hc.parse_reply(_reply("jawaban").decode("utf-8")), "jawaban")

    def test_rejects_malformed_payloads(self):
        bad_payloads = (
            b"not json",
            json.dumps({"choices": []}),
            json.dumps({"choices": [{"message": {"content": ""}}]}),
            json.dumps({"error": "boom"}),
        )
        for raw in bad_payloads:
            with self.assertRaises(RuntimeError):
                hc.parse_reply(raw)


class ChatTest(unittest.TestCase):
    def test_request_targets_model_and_budget(self):
        seen = {}

        def transport(url, payload):
            seen["url"] = url
            seen["payload"] = payload
            return _reply("Istirahat dan minum air cukup.")

        text = hc.chat("Demam?", base_url="http://localhost:9999/v1/", transport=transport)
        self.assertEqual(text, "Istirahat dan minum air cukup.")
        self.assertEqual(seen["url"], "http://localhost:9999/v1/chat/completions")
        self.assertEqual(seen["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(seen["payload"]["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(seen["payload"]["max_tokens"], 1024)
        self.assertEqual(seen["payload"]["messages"][-1]["content"], "Demam?")

    def test_token_budget_override(self):
        seen = {}

        def transport(url, payload):
            seen["payload"] = payload
            return _reply("ok")

        hc.chat("Demam?", transport=transport, max_new_tokens=64)
        self.assertEqual(seen["payload"]["max_tokens"], 64)

    def test_unreachable_endpoint_is_explained(self):
        def transport(url, payload):
            raise OSError("connection refused")

        with self.assertRaises(RuntimeError) as ctx:
            hc.chat("Demam?", transport=transport)
        self.assertIn("SPARK_BASE_URL", str(ctx.exception))


class AnswerTest(unittest.TestCase):
    def test_disclaimer_added_once(self):
        plain = lambda url, payload: _reply("Istirahat dan minum air.")
        text = hc.answer("Demam?", transport=plain)
        self.assertTrue(text.startswith("Istirahat dan minum air."))
        self.assertEqual(text.count(hc.DISCLAIMER), 1)

        echoed = lambda url, payload: _reply("Istirahat.\n\n" + hc.DISCLAIMER)
        self.assertEqual(hc.answer("Demam?", transport=echoed).count(hc.DISCLAIMER), 1)

    def test_demo_question_uses_model_path(self):
        seen = {}

        def transport(url, payload):
            seen["payload"] = payload
            return _reply("Perhatikan demam tinggi atau sesak napas.")

        hc.answer(hc.DEMO_QUESTION, transport=transport)
        self.assertEqual(seen["payload"]["messages"][-1]["content"], hc.DEMO_QUESTION.strip())


if __name__ == "__main__":
    unittest.main()
