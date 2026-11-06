"""Offline tests for the Spark-X2.6-4B health chatbot helpers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = ["<%s>%s</%s>" % (m["role"], m["content"], m["role"]) for m in messages]
        return "".join(parts) + ("<assistant>" if add_generation_prompt else "")


class ConfigTest(unittest.TestCase):
    def test_model_id(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_max_new_tokens(self):
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)


class SystemPromptTest(unittest.TestCase):
    def test_forbids_diagnosis(self):
        self.assertIn("never provide a diagnosis", hc.SYSTEM_PROMPT)

    def test_mentions_urgent_care(self):
        self.assertIn("urgent care", hc.SYSTEM_PROMPT)
        self.assertIn("chest pain", hc.SYSTEM_PROMPT)

    def test_mentions_language_and_privacy(self):
        self.assertIn("answer in the language", hc.SYSTEM_PROMPT)
        self.assertIn("identifying details", hc.SYSTEM_PROMPT)


class BuildMessagesTest(unittest.TestCase):
    def test_system_first_user_last(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_question_is_stripped(self):
        messages = hc.build_messages("  hello  ")
        self.assertEqual(messages[-1]["content"], "hello")

    def test_history_filters_and_preserves_order(self):
        history = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": " earlier "},
            {"role": "assistant", "content": "reply"},
            {"role": "tool", "content": "ignored"},
        ]
        messages = hc.build_messages("next", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1], {"role": "user", "content": "earlier"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "reply"})

    def test_empty_question_still_returns_user_turn(self):
        messages = hc.build_messages(None)
        self.assertEqual(messages[-1], {"role": "user", "content": ""})


class FormatPromptTest(unittest.TestCase):
    def test_uses_tokenizer_template(self):
        prompt = hc.format_prompt(FakeTokenizer(), hc.build_messages("hi"))
        self.assertTrue(prompt.startswith("<system>"))
        self.assertTrue(prompt.endswith("<assistant>"))

    def test_fallback_without_tokenizer(self):
        prompt = hc.format_prompt(None, hc.build_messages("hi"))
        self.assertIn("SYSTEM:", prompt)
        self.assertIn("USER: hi", prompt)
        self.assertTrue(prompt.endswith("ASSISTANT:"))


class UrgentSymptomTest(unittest.TestCase):
    def test_detects_red_flag(self):
        self.assertTrue(hc.has_urgent_symptom("I have chest pain since morning"))

    def test_ignores_common_symptom(self):
        self.assertFalse(hc.has_urgent_symptom("I feel tired after work"))

    def test_handles_empty_text(self):
        self.assertFalse(hc.has_urgent_symptom(""))
        self.assertFalse(hc.has_urgent_symptom(None))


class AnswerPostProcessingTest(unittest.TestCase):
    def test_clean_response_strips_whitespace(self):
        self.assertEqual(hc.clean_response("  answer  "), "answer")

    def test_strip_prompt_removes_echo(self):
        prompt = "SYSTEM: x\n\nUSER: hi\n\nASSISTANT:"
        text = prompt + " drink water"
        self.assertEqual(hc.strip_prompt(text, prompt), "drink water")

    def test_safety_note_appended_when_missing(self):
        result = hc.ensure_safety_note("Rest and drink water.")
        self.assertIn("Rest and drink water.", result)
        self.assertIn(hc.SAFETY_NOTE, result)

    def test_safety_note_not_duplicated(self):
        answer = "Please see a doctor about this."
        self.assertEqual(hc.ensure_safety_note(answer), answer)

    def test_finalize_answer_combines_steps(self):
        result = hc.finalize_answer("  Take care of yourself.  ")
        self.assertTrue(result.startswith("Take care of yourself."))
        self.assertIn(hc.SAFETY_NOTE, result)


class ChatbotInitTest(unittest.TestCase):
    def test_model_is_not_loaded_on_construction(self):
        bot = hc.HealthChatbot()
        self.assertIsNone(bot._generator)
        self.assertEqual(bot.model_id, hc.MODEL_ID)
        self.assertEqual(bot.max_new_tokens, hc.MAX_NEW_TOKENS)


if __name__ == "__main__":
    unittest.main()
