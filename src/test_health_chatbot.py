"""Tests for the Spark-X2.6-4B health-information chatbot.

Run from the repository root:

    python -m unittest discover -s src -p "test_*.py"

No model weights, GPU or third-party packages are required.
"""

import io
import json
import unittest
from contextlib import redirect_stdout

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("hello")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})

    def test_question_is_the_last_user_turn(self):
        messages = hc.build_messages("  fever for two days  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "fever for two days"})

    def test_history_is_kept_before_the_question(self):
        history = [
            {"role": "user", "content": "I am 30 years old."},
            {"role": "assistant", "content": "Thanks for telling me."},
        ]
        messages = hc.build_messages("And now?", history=history)
        self.assertEqual(
            [m["content"] for m in messages[1:-1]],
            ["I am 30 years old.", "Thanks for telling me."],
        )
        self.assertEqual(messages[-1]["content"], "And now?")

    def test_history_is_not_mutated(self):
        history = [{"role": "user", "content": "hi"}]
        hc.build_messages("again", history=history)
        self.assertEqual(history, [{"role": "user", "content": "hi"}])

    def test_blank_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_unknown_history_role_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("q", history=[{"role": "tool", "content": "x"}])


class BuildPromptTest(unittest.TestCase):
    def test_fallback_prompt_contains_all_turns(self):
        prompt = hc.build_prompt(hc.build_messages("cough"))
        self.assertIn(hc.SYSTEM_PROMPT, prompt)
        self.assertIn("cough", prompt)
        self.assertTrue(prompt.rstrip().endswith("<|assistant|>"))

    def test_tokenizer_chat_template_is_used_when_available(self):
        class FakeTokenizer:
            def apply_chat_template(
                self, messages, tokenize=False, add_generation_prompt=True
            ):
                return "TEMPLATED:" + messages[-1]["content"]

        prompt = hc.build_prompt(hc.build_messages("cough"), tokenizer=FakeTokenizer())
        self.assertEqual(prompt, "TEMPLATED:cough")


class SafetyTest(unittest.TestCase):
    def test_disclaimer_is_added_once(self):
        once = hc.ensure_disclaimer("Rest and drink water.")
        self.assertIn(hc.DISCLAIMER, once)
        self.assertEqual(hc.ensure_disclaimer(once), once)

    def test_empty_reply_still_gets_disclaimer(self):
        self.assertEqual(hc.ensure_disclaimer(""), hc.DISCLAIMER)

    def test_system_prompt_forbids_diagnosis(self):
        self.assertIn("do NOT diagnose", hc.SYSTEM_PROMPT)
        self.assertIn("emergency", hc.SYSTEM_PROMPT.lower())


class ExtractReplyTest(unittest.TestCase):
    def test_chat_style_output(self):
        output = [
            {
                "generated_text": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        ]
        self.assertEqual(hc.extract_reply(output), "a")

    def test_plain_string_output(self):
        self.assertEqual(hc.extract_reply([{"generated_text": "plain"}]), "plain")

    def test_empty_output(self):
        self.assertEqual(hc.extract_reply([]), "")


class MainTest(unittest.TestCase):
    def test_dry_run_prints_prompt_without_model(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = hc.main(["--dry-run", "sore throat"])
        self.assertEqual(code, 0)
        self.assertIn("sore throat", buffer.getvalue())

    def test_json_dry_run_is_valid_json(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            hc.main(["--dry-run", "--json", "sore throat"])
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertIsNone(payload["reply"])


if __name__ == "__main__":
    unittest.main()
