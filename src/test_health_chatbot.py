"""Tests for the minimal Spark-X2.6-4B health-education chatbot.

No network and no third-party packages: the model call is exercised with
an injected fake client.
"""

import unittest
from types import SimpleNamespace

import health_chatbot as hc


class _FakeClient:
    def __init__(self, content):
        self.calls = []
        self._content = content
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class ConstantsTest(unittest.TestCase):
    def test_targets_spark_x2_6_4b(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_forbids_diagnosis_and_prescription(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never give a diagnosis", prompt)
        self.assertIn("medicine", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("emergency", prompt)

    def test_safety_notice_is_code_side(self):
        self.assertIn("not a diagnosis", hc.SAFETY_NOTICE)
        self.assertNotIn(hc.SAFETY_NOTICE, hc.SYSTEM_PROMPT)


class BuildMessagesTest(unittest.TestCase):
    def test_system_then_user(self):
        messages = hc.build_messages("  what is a fever?  ")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1], {"role": "user", "content": "what is a fever?"})

    def test_history_is_kept_in_order_and_not_mutated(self):
        history = [
            {"role": "user", "content": "I have a cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        snapshot = [dict(turn) for turn in history]
        messages = hc.build_messages("Three days.", history=history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[-1]["content"], "Three days.")
        self.assertEqual(history, snapshot)

    def test_rejects_blank_question(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_rejects_bad_history(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hi", history=[{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("hi", history=[{"role": "user", "content": ""}])


class GenerateReplyTest(unittest.TestCase):
    def test_sends_model_and_token_budget(self):
        client = _FakeClient("  Rest, fluids, and watch for red flags.  ")
        reply = hc.generate_reply("what helps a fever?", client=client)
        self.assertEqual(reply, "Rest, fluids, and watch for red flags.")
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        self.assertEqual(call["model"], hc.MODEL_ID)
        self.assertEqual(call["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(call["messages"][0]["role"], "system")
        self.assertEqual(call["messages"][-1]["content"], "what helps a fever?")

    def test_empty_model_output_is_an_error(self):
        client = _FakeClient("   ")
        with self.assertRaises(RuntimeError):
            hc.generate_reply("anything", client=client)

    def test_demo_question_reaches_the_model(self):
        client = _FakeClient("ok")
        hc.generate_reply(hc.DEMO_QUESTION, client=client)
        self.assertEqual(client.calls[0]["messages"][-1]["content"], hc.DEMO_QUESTION)

    def test_answer_always_attaches_safety_notice(self):
        client = _FakeClient("Some explanation.")
        result = hc.answer(hc.DEMO_QUESTION, client=client)
        self.assertEqual(result["model"], hc.MODEL_ID)
        self.assertEqual(result["answer"], "Some explanation.")
        self.assertEqual(result["safety_notice"], hc.SAFETY_NOTICE)


if __name__ == "__main__":
    unittest.main()
