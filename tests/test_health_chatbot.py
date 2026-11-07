"""Unit tests for src/health_chatbot.py (no network, no extra dependencies)."""

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hb  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hb.build_messages("What is hypertension?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hb.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is hypertension?"})

    def test_system_prompt_blocks_diagnosis(self):
        prompt = hb.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("never claim to replace a doctor", prompt)

    def test_blank_question_is_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                hb.build_messages(bad)

    def test_question_is_stripped(self):
        messages = hb.build_messages("  fever for three days  ")
        self.assertEqual(messages[-1]["content"], "fever for three days")

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
            {"role": "tool", "content": "ignored"},
        ]
        messages = hb.build_messages("Two weeks.", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "I have a cough.")

    def test_demo_question_is_available(self):
        self.assertTrue(hb.DEMO_QUESTION.strip())
        self.assertEqual(hb.build_messages(hb.DEMO_QUESTION)[-1]["content"], hb.DEMO_QUESTION)


class PayloadTest(unittest.TestCase):
    def test_payload_targets_spark_model(self):
        messages = hb.build_messages("What is anaemia?")
        payload = hb.build_payload(messages)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["model"], hb.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hb.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"], messages)

    def test_payload_does_not_alias_input(self):
        messages = hb.build_messages("What is anaemia?")
        payload = hb.build_payload(messages)
        payload["messages"][0]["content"] = "changed"
        self.assertEqual(messages[0]["content"], hb.SYSTEM_PROMPT)


class ParseResponseTest(unittest.TestCase):
    def test_extracts_and_strips_content(self):
        data = {"choices": [{"message": {"content": "  Ask your clinician.  "}}]}
        self.assertEqual(hb.parse_response(data), "Ask your clinician.")

    def test_rejects_missing_choices(self):
        with self.assertRaises(ValueError):
            hb.parse_response({})

    def test_rejects_empty_content(self):
        data = {"choices": [{"message": {"content": "   "}}]}
        with self.assertRaises(ValueError):
            hb.parse_response(data)


class AskTest(unittest.TestCase):
    def test_ask_calls_endpoint_and_returns_answer(self):
        seen = {}

        def fake_opener(request, timeout=None):
            seen["url"] = request.full_url
            seen["timeout"] = timeout
            seen["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({"choices": [{"message": {"content": "Ask your doctor."}}]})

        answer = hb.ask(
            "dry cough after travel",
            base_url="http://127.0.0.1:30000/v1/",
            timeout=7,
            opener=fake_opener,
        )

        self.assertEqual(answer, "Ask your doctor.")
        self.assertEqual(seen["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(seen["timeout"], 7)
        self.assertEqual(seen["payload"]["model"], hb.MODEL_ID)
        self.assertEqual(seen["payload"]["messages"][-1]["content"], "dry cough after travel")

    def test_ask_propagates_parse_errors(self):
        def broken_opener(request, timeout=None):
            return FakeResponse({"choices": []})

        with self.assertRaises(ValueError):
            hb.ask("fever", opener=broken_opener)


if __name__ == "__main__":
    unittest.main()
