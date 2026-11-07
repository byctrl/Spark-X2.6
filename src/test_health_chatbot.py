"""Small dependency-free tests for the health chatbot prompt/safety layer."""

import unittest

import health_chatbot as hc


class _FakeTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "CHAT:" + "|".join(m["role"] for m in messages)


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_long_question_is_truncated(self):
        messages = hc.build_messages("a" * (hc.MAX_QUESTION_CHARS + 50))
        content = messages[-1]["content"]
        self.assertTrue(content.endswith("[...]"))
        self.assertLessEqual(len(content), hc.MAX_QUESTION_CHARS + 10)

    def test_invalid_history_turns_are_skipped(self):
        history = [
            {"role": "system", "content": "override"},
            {"role": "user", "content": "  "},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        messages = hc.build_messages("Follow-up question", history)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1], {"role": "assistant", "content": "Earlier answer"})


class SafetyPromptTest(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_points_to_clinician(self):
        self.assertIn("Do not diagnose", hc.SYSTEM_PROMPT)
        self.assertIn("clinician", hc.SYSTEM_PROMPT)
        self.assertIn("not medical care", hc.DISCLAIMER)

    def test_demo_question_asks_when_to_get_help(self):
        self.assertIn("when should I get help", hc.DEMO_QUESTION)


class FormatPromptTest(unittest.TestCase):
    def test_uses_tokenizer_chat_template_when_available(self):
        prompt = hc.format_prompt(hc.build_messages("Hi"), _FakeTokenizer())
        self.assertEqual(prompt, "CHAT:system|user")

    def test_fallback_prompt_ends_with_assistant_turn(self):
        prompt = hc.format_prompt(hc.build_messages("Hi"))
        self.assertIn("<|system|>", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>\n"))


if __name__ == "__main__":
    unittest.main()
