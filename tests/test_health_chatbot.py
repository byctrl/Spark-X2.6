"""Unit tests for the Spark-X2.6-4B health-information chatbot (no network)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import health_chatbot as hc  # noqa: E402


class FakeTransport:
    """Stand-in for the HTTP transport; records the payload it received."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, base_url, payload):
        self.calls.append((base_url, payload))
        return {"choices": [{"message": {"content": self.reply}}]}


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_is_preserved_and_filtered(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("next", history)
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])


class ParseReplyTest(unittest.TestCase):
    def test_reads_content(self):
        data = {"choices": [{"message": {"content": "  plain answer  "}}]}
        self.assertEqual(hc.parse_reply(data), "plain answer")

    def test_rejects_bad_shape(self):
        with self.assertRaises(RuntimeError):
            hc.parse_reply({"choices": []})

    def test_rejects_empty_content(self):
        with self.assertRaises(RuntimeError):
            hc.parse_reply({"choices": [{"message": {"content": "  "}}]})


class DisclaimerTest(unittest.TestCase):
    def test_appended_once(self):
        once = hc.with_disclaimer("Some education.")
        self.assertIn(hc.DISCLAIMER, once)
        self.assertEqual(hc.with_disclaimer(once), once)


class HealthChatbotTest(unittest.TestCase):
    def test_payload_uses_model_and_token_budget(self):
        transport = FakeTransport("Some education.")
        bot = hc.HealthChatbot(base_url="http://localhost:30000/v1", transport=transport)
        bot.ask("What is a fever?")

        base_url, payload = transport.calls[0]
        self.assertEqual(base_url, "http://localhost:30000/v1")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "system")

    def test_answer_carries_disclaimer_and_history(self):
        transport = FakeTransport("Coughs are common after travel. Seek care if breathing is hard.")
        bot = hc.HealthChatbot(transport=transport)
        answer = bot.ask(hc.DEMO_QUESTION)

        self.assertIn("Coughs are common after travel.", answer)
        self.assertIn(hc.DISCLAIMER, answer)
        self.assertEqual(len(bot.history), 2)
        self.assertEqual(bot.history[0]["role"], "user")
        self.assertEqual(bot.history[1]["role"], "assistant")

        bot.ask("And what about the fever?")
        second_messages = transport.calls[1][1]["messages"]
        self.assertEqual(second_messages[0]["role"], "system")
        self.assertEqual(second_messages[-1]["content"], "And what about the fever?")
        self.assertEqual(len(second_messages), 4)

    def test_demo_question_is_a_symptom_question(self):
        self.assertIn("fever", hc.DEMO_QUESTION.lower())


if __name__ == "__main__":
    unittest.main()
