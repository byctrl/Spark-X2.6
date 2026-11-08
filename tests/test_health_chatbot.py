"""Unit tests for the Spark-X2.6 health-information chatbot demo."""

import unittest

from src import health_chatbot as hc


class BuildMessagesTests(unittest.TestCase):
    def test_includes_system_prompt_and_question(self):
        messages = hc.build_messages("  hello  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "hello"})

    def test_history_turns_are_preserved(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "ignored"},
            {"content": "no role"},
            "not a dict",
        ]
        messages = hc.build_messages("q", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[-1]["content"], "q")

    def test_empty_question_is_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_system_prompt_forbids_diagnosis(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        self.assertIn("do not diagnose", lowered)
        self.assertIn("clinician", lowered)


class FormatPromptTests(unittest.TestCase):
    def test_prompt_contains_question_and_assistant_prefix(self):
        prompt = hc.format_prompt(hc.build_messages("cough and fever"))
        self.assertIn("cough and fever", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertEqual(prompt.count("<|im_start|>system"), 1)


class DisclaimerTests(unittest.TestCase):
    def test_disclaimer_is_appended_once(self):
        self.assertEqual(hc.ensure_disclaimer("text"), "text\n\n" + hc.DISCLAIMER)
        self.assertEqual(hc.ensure_disclaimer(hc.DISCLAIMER), hc.DISCLAIMER)

    def test_empty_text_returns_disclaimer(self):
        self.assertEqual(hc.ensure_disclaimer(""), hc.DISCLAIMER)
        self.assertEqual(hc.ensure_disclaimer(None), hc.DISCLAIMER)


class ChatbotTests(unittest.TestCase):
    def test_generate_uses_injected_generator(self):
        seen = {}

        def fake_generator(prompt):
            seen["prompt"] = prompt
            return "Rest and see a doctor.<|im_end|>"

        bot = hc.SparkHealthChatbot(generator=fake_generator)
        answer = bot.generate(hc.DEMO_QUESTION)

        self.assertIn("Rest and see a doctor.", answer)
        self.assertNotIn("<|im_end|>", answer)
        self.assertTrue(answer.endswith(hc.DISCLAIMER))
        self.assertIn(hc.DEMO_QUESTION, seen["prompt"])

    def test_disclaimer_not_duplicated_when_model_emits_it(self):
        bot = hc.SparkHealthChatbot(generator=lambda prompt: hc.DISCLAIMER)
        self.assertEqual(bot.generate("q").count(hc.DISCLAIMER), 1)

    def test_empty_model_output_still_returns_disclaimer(self):
        bot = hc.SparkHealthChatbot(generator=lambda prompt: "")
        self.assertEqual(bot.generate("q"), hc.DISCLAIMER)

    def test_history_is_forwarded_to_the_model(self):
        seen = {}

        def fake_generator(prompt):
            seen["prompt"] = prompt
            return "ok"

        bot = hc.SparkHealthChatbot(generator=fake_generator)
        bot.generate("follow up", [{"role": "user", "content": "earlier turn"}])
        self.assertIn("earlier turn", seen["prompt"])
        self.assertIn("follow up", seen["prompt"])

    def test_defaults_match_the_released_model(self):
        bot = hc.SparkHealthChatbot()
        self.assertEqual(bot.model_id, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertGreater(bot.max_new_tokens, 0)
        self.assertLessEqual(bot.max_new_tokens, 4096)

    def test_pipeline_is_not_loaded_when_generator_is_injected(self):
        bot = hc.SparkHealthChatbot(generator=lambda prompt: "ok")
        bot.generate("q")
        self.assertIsNone(bot._pipeline)


if __name__ == "__main__":
    unittest.main()
