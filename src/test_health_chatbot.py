"""Offline tests for src/health_chatbot.py (no model download required)."""

from __future__ import annotations

import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self) -> None:
        messages = hc.build_messages("Apa itu demam?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Apa itu demam?"})

    def test_question_is_stripped(self) -> None:
        messages = hc.build_messages("   batuk kering   ")
        self.assertEqual(messages[-1]["content"], "batuk kering")

    def test_empty_question_is_rejected(self) -> None:
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)  # type: ignore[arg-type]

    def test_history_is_filtered(self) -> None:
        history = [
            {"role": "user", "content": "Halo"},
            {"role": "assistant", "content": "Halo, ada yang bisa dibantu?"},
            {"role": "system", "content": "ignored"},
            {"role": "tool", "content": "ignored"},
            {"role": "user", "content": "   "},
        ]
        messages = hc.build_messages("Saya demam.", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "Saya demam.")

    def test_system_prompt_sets_safety_boundaries(self) -> None:
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("emergency", prompt)


class RenderPromptTest(unittest.TestCase):
    def test_fallback_rendering_contains_all_turns(self) -> None:
        messages = hc.build_messages("Apa itu demam?")
        prompt = hc.render_prompt(messages)
        self.assertIn("SYSTEM: ", prompt)
        self.assertIn("USER: Apa itu demam?", prompt)

    def test_chat_template_is_used_when_available(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                self.called = True
                return "templated:{0}".format(len(messages))

        tokenizer = FakeTokenizer()
        messages = hc.build_messages("Apa itu demam?")
        self.assertEqual(hc.render_prompt(messages, tokenizer), "templated:2")
        self.assertTrue(tokenizer.called)


class GenerateReplyTest(unittest.TestCase):
    def test_uses_injected_generator(self) -> None:
        seen = {}

        def fake_generator(messages):
            seen["messages"] = messages
            return "  Demam ringan biasanya membaik dengan istirahat.  "

        reply = hc.generate_reply("Saya demam.", generator=fake_generator)
        self.assertEqual(reply, "Demam ringan biasanya membaik dengan istirahat.")
        self.assertEqual(seen["messages"][0]["role"], "system")
        self.assertEqual(seen["messages"][-1]["content"], "Saya demam.")

    def test_history_is_passed_to_generator(self) -> None:
        seen = {}

        def fake_generator(messages):
            seen["messages"] = list(messages)
            return "ok"

        history = [{"role": "user", "content": "Halo"}, {"role": "assistant", "content": "Halo"}]
        hc.generate_reply("Saya demam.", history=history, generator=fake_generator)
        self.assertEqual(len(seen["messages"]), 4)

    def test_empty_reply_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            hc.generate_reply("Saya demam.", generator=lambda messages: "   ")

    def test_none_reply_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            hc.generate_reply("Saya demam.", generator=lambda messages: None)  # type: ignore[return-value]


class FormatAnswerTest(unittest.TestCase):
    def test_disclaimer_is_appended(self) -> None:
        answer = hc.format_answer("  Istirahat yang cukup.  ")
        self.assertTrue(answer.startswith("Istirahat yang cukup."))
        self.assertIn(hc.DISCLAIMER, answer)

    def test_demo_question_is_set(self) -> None:
        self.assertIn("batuk", hc.DEMO_QUESTION.lower())
        self.assertIn("demam", hc.DEMO_QUESTION.lower())


def test_model_constants() -> None:
    assert hc.MODEL_ID == "XHToken/Spark-X2.6-4B"
    assert hc.MAX_NEW_TOKENS == 1024


if __name__ == "__main__":
    unittest.main()
