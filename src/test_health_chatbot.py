"""Offline tests for the Spark-X2.6-4B health-information chatbot."""

import unittest
from unittest import mock

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_and_question(self):
        messages = hc.build_messages("  What is hypertension?  ")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(
            messages[1],
            {"role": "user", "content": "What is hypertension?"},
        )

    def test_history_is_kept_in_order_and_invalid_turns_dropped(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("next", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[-1]["content"], "next")


class SafetyTest(unittest.TestCase):
    def test_system_prompt_forbids_diagnosis(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("not medical advice", prompt)

    def test_emergency_text_is_detected(self):
        for text in ("I have chest pain", "I cannot breathe", "\u80f8\u75db\u600e\u4e48\u529e"):
            self.assertEqual(hc.emergency_notice(text), hc.EMERGENCY_NOTICE)

    def test_ordinary_text_is_not_an_emergency(self):
        self.assertIsNone(hc.emergency_notice("What is a normal sleep schedule?"))

    def test_emergency_short_circuits_transport(self):
        calls = []

        def transport(messages):
            calls.append(messages)
            return "should not run"

        reply = hc.answer("crushing chest pain right now", transport=transport)
        self.assertEqual(reply, hc.EMERGENCY_NOTICE)
        self.assertEqual(calls, [])


class AnswerTest(unittest.TestCase):
    def test_empty_question(self):
        self.assertIn("Please type", hc.answer("   "))

    def test_reply_includes_disclaimer(self):
        seen = {}

        def transport(messages):
            seen["messages"] = messages
            return "  Blood pressure is the force of blood against artery walls.  "

        reply = hc.answer("What is blood pressure?", transport=transport)
        self.assertTrue(reply.startswith("Blood pressure is the force"))
        self.assertTrue(reply.endswith(hc.DISCLAIMER))
        self.assertEqual(seen["messages"][-1]["content"], "What is blood pressure?")

    def test_empty_model_reply_uses_fallback(self):
        reply = hc.answer("What is blood pressure?", transport=lambda messages: "   ")
        self.assertIn(hc.FALLBACK, reply)
        self.assertIn(hc.DISCLAIMER, reply)

    def test_default_transport_posts_to_spark_endpoint(self):
        captured = {}

        def fake_post(url, payload):
            captured["url"] = url
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "ok"}}]}

        with mock.patch.object(hc, "_post_json", side_effect=fake_post):
            text = hc.default_transport(hc.build_messages("hi"))

        self.assertEqual(text, "ok")
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(captured["payload"]["max_tokens"], hc.MAX_NEW_TOKENS)

    def test_default_transport_reports_bad_response(self):
        with mock.patch.object(hc, "_post_json", return_value={}):
            with self.assertRaises(RuntimeError):
                hc.default_transport(hc.build_messages("hi"))


if __name__ == "__main__":
    unittest.main()
