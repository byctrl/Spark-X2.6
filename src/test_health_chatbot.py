"""Dependency-free tests for src/health_chatbot.py.

No model weights are downloaded: generation is injected with a stub.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class TestConfig(unittest.TestCase):
    def test_model_id(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_max_new_tokens_is_bounded(self):
        self.assertIsInstance(hc.MAX_NEW_TOKENS, int)
        self.assertGreater(hc.MAX_NEW_TOKENS, 0)
        self.assertLessEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_demo_question_present(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


class TestSystemPromptSafety(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_prescription(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never give a diagnosis", lowered)
        self.assertIn("prescribe", lowered)

    def test_prompt_points_to_clinician(self):
        self.assertIn("clinician", hc.SYSTEM_PROMPT.lower())

    def test_prompt_forbids_identifying_data(self):
        self.assertIn("identifying data", hc.SYSTEM_PROMPT.lower())

    def test_emergency_note_mentions_emergency_services(self):
        self.assertIn("emergency services", hc.EMERGENCY_NOTE.lower())


class TestBuildMessages(unittest.TestCase):
    def test_system_first_user_last(self):
        messages = hc.build_messages("  what is a fever?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "what is a fever?"})

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi, how can I help?"},
        ]
        messages = hc.build_messages("and now?", history=history)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])

    def test_history_is_truncated(self):
        history = [
            {"role": "user", "content": "turn %d" % i}
            for i in range(hc.MAX_HISTORY_TURNS + 4)
        ]
        messages = hc.build_messages("last?", history=history)
        self.assertEqual(len(messages), hc.MAX_HISTORY_TURNS + 2)

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_non_string_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages(None)

    def test_bad_history_role_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("q", history=[{"role": "system", "content": "x"}])

    def test_empty_history_content_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("q", history=[{"role": "user", "content": ""}])


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.seen = messages
        return "TEMPLATE"


class TestFormatPrompt(unittest.TestCase):
    def test_fallback_transcript(self):
        prompt = hc.format_prompt(hc.build_messages("hello"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("hello", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>\n"))

    def test_uses_chat_template_when_available(self):
        tokenizer = _FakeTokenizer()
        prompt = hc.format_prompt(hc.build_messages("hello"), tokenizer=tokenizer)
        self.assertEqual(prompt, "TEMPLATE")
        self.assertEqual(tokenizer.seen[0]["role"], "system")

    def test_empty_messages_rejected(self):
        with self.assertRaises(ValueError):
            hc.format_prompt([])


class TestCleanReply(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(hc.clean_reply("  hello  \n"), "hello")

    def test_cuts_stop_markers(self):
        self.assertEqual(hc.clean_reply("answer<|end|>junk"), "answer")
        self.assertEqual(hc.clean_reply("answer<|user|>next"), "answer")

    def test_rejects_non_string(self):
        with self.assertRaises(ValueError):
            hc.clean_reply(None)


class TestGenerateReply(unittest.TestCase):
    def test_uses_injected_chat_fn(self):
        captured = {}

        def fake_chat(messages):
            captured["messages"] = messages
            return "  Rest, fluids, and see a clinician if fever lasts. <|end|>"

        result = hc.generate_reply("I have a cough", chat_fn=fake_chat)
        self.assertEqual(result["reply"],
                         "Rest, fluids, and see a clinician if fever lasts.")
        self.assertEqual(result["model"], hc.MODEL_ID)
        self.assertIn("emergency_note", result)
        self.assertEqual(captured["messages"][0]["role"], "system")
        self.assertEqual(captured["messages"][-1]["content"], "I have a cough")

    def test_empty_model_reply_raises(self):
        with self.assertRaises(RuntimeError):
            hc.generate_reply("q", chat_fn=lambda messages: "   ")

    def test_invalid_question_raises_before_model_call(self):
        def exploding_chat(messages):
            raise AssertionError("model must not be called")

        with self.assertRaises(ValueError):
            hc.generate_reply("", chat_fn=exploding_chat)


if __name__ == "__main__":
    unittest.main()
