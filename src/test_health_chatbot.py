"""Offline unit tests for src/health_chatbot.py (no model server needed)."""

import json
import unittest

import health_chatbot as hc


class FakeTransport(object):
    def __init__(self, content="Rest and drink fluids.", finish_reason="stop"):
        self.content = content
        self.finish_reason = finish_reason
        self.calls = []

    def __call__(self, url, body, timeout):
        self.calls.append({"url": url, "body": body, "timeout": timeout})
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": self.content},
                    "finish_reason": self.finish_reason,
                }
            ]
        }


def failing_transport(url, body, timeout):
    raise AssertionError("network transport must not be used in tests")


class ValidateQuestionTests(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(hc.validate_question("  hi  "), "hi")

    def test_rejects_empty(self):
        for value in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.validate_question(value)

    def test_rejects_too_long(self):
        with self.assertRaises(ValueError):
            hc.validate_question("a" * (hc.MAX_QUESTION_CHARS + 1))


class BuildMessagesTests(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("What is asthma?")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "What is asthma?"})
        self.assertEqual(len(messages), 2)

    def test_history_is_kept_in_order(self):
        history = [{"role": "user", "content": "q1"},
                   {"role": "assistant", "content": "a1"}]
        messages = hc.build_messages("q2", history=history)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])

    def test_bad_history_entry_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("q", history=[{"role": "system", "content": "x"}])


class BuildPayloadTests(unittest.TestCase):
    def test_uses_model_id_and_token_budget(self):
        payload = hc.build_payload(hc.build_messages("q"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertIn("messages", payload)

    def test_rejects_empty_messages(self):
        with self.assertRaises(ValueError):
            hc.build_payload([])


class ExtractTextTests(unittest.TestCase):
    def test_extracts_and_strips(self):
        data = {"choices": [{"message": {"content": "  hello  "}}]}
        self.assertEqual(hc.extract_text(data), "hello")

    def test_rejects_empty_content(self):
        data = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        with self.assertRaises(ValueError) as ctx:
            hc.extract_text(data)
        self.assertIn("length", str(ctx.exception))

    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            hc.extract_text({"error": "boom"})


class UrgencyTests(unittest.TestCase):
    def test_demo_question_is_not_flagged(self):
        self.assertFalse(hc.is_urgent(hc.DEMO_QUESTION))

    def test_red_flag_is_flagged_case_insensitively(self):
        self.assertTrue(hc.is_urgent("I have Chest Pain since this morning"))


class AnswerTests(unittest.TestCase):
    def test_hits_chat_completions_endpoint(self):
        transport = FakeTransport()
        hc.answer("What is asthma?",
                  base_url="http://127.0.0.1:30000/v1", transport=transport)
        self.assertEqual(transport.calls[0]["url"],
                         "http://127.0.0.1:30000/v1/chat/completions")
        payload = json.loads(transport.calls[0]["body"].decode("utf-8"))
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["messages"][-1]["content"], "What is asthma?")

    def test_appends_safety_footer(self):
        result = hc.answer("What is asthma?", transport=FakeTransport("Asthma info."))
        self.assertTrue(result.startswith("Asthma info."))
        self.assertIn("not a diagnosis", result)
        self.assertNotIn(hc.URGENT_BANNER, result)

    def test_prepends_urgent_banner_for_red_flags(self):
        result = hc.answer("I cannot breathe well", transport=FakeTransport("Call someone."))
        self.assertTrue(result.startswith(hc.URGENT_BANNER))
        self.assertIn("not a diagnosis", result)

    def test_invalid_question_never_calls_model(self):
        with self.assertRaises(ValueError):
            hc.answer("   ", transport=failing_transport)

    def test_default_transport_not_used_when_injected(self):
        transport = FakeTransport()
        hc.answer(hc.DEMO_QUESTION, transport=transport)
        self.assertEqual(len(transport.calls), 1)


if __name__ == "__main__":
    unittest.main()
