"""Dependency-free tests for the Spark-X2.6 health chatbot demo.

Run from the repository root:

    python src/test_health_chatbot.py
"""

import unittest

from health_chatbot import (
    DEMO_QUESTION,
    DISCLAIMER,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    build_messages,
    finalize_reply,
    generate_reply,
    render_prompt,
)


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_and_question(self):
        messages = build_messages("  " + DEMO_QUESTION + "  ")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[1], {"role": "user", "content": DEMO_QUESTION})

    def test_rejects_invalid_question(self):
        for bad in ("", "   ", None, 5):
            with self.assertRaises(ValueError):
                build_messages(bad)

    def test_system_prompt_forbids_diagnosis(self):
        self.assertIn("Do NOT give a diagnosis", SYSTEM_PROMPT)


class RenderPromptTest(unittest.TestCase):
    def test_contains_system_and_question(self):
        prompt = render_prompt(build_messages("I have a fever."))
        self.assertIn(SYSTEM_PROMPT, prompt)
        self.assertIn("I have a fever.", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_rejects_empty_messages(self):
        with self.assertRaises(ValueError):
            render_prompt([])


class FinalizeReplyTest(unittest.TestCase):
    def test_appends_disclaimer(self):
        reply = finalize_reply("  Rest and drink water.  ")
        self.assertTrue(reply.startswith("Rest and drink water."))
        self.assertIn(DISCLAIMER, reply)

    def test_disclaimer_not_duplicated(self):
        reply = finalize_reply("Rest.\n\n" + DISCLAIMER)
        self.assertEqual(reply.count(DISCLAIMER), 1)

    def test_empty_output_is_safe(self):
        reply = finalize_reply("   ")
        self.assertIn("could not produce a safe answer", reply)
        self.assertIn(DISCLAIMER, reply)

    def test_rejects_non_string(self):
        with self.assertRaises(ValueError):
            finalize_reply(None)


class GenerateReplyTest(unittest.TestCase):
    def test_uses_generator_with_model_budget(self):
        seen = {}

        def fake_generator(prompt, max_new_tokens):
            seen["prompt"] = prompt
            seen["max_new_tokens"] = max_new_tokens
            return "  Seek care if breathing becomes difficult.  "

        reply = generate_reply("I have a fever.", generator=fake_generator)
        self.assertIn("I have a fever.", seen["prompt"])
        self.assertIn(SYSTEM_PROMPT, seen["prompt"])
        self.assertEqual(seen["max_new_tokens"], MAX_NEW_TOKENS)
        self.assertTrue(reply.startswith("Seek care if breathing"))
        self.assertIn(DISCLAIMER, reply)

    def test_model_id_is_spark(self):
        self.assertEqual(MODEL_ID, "XHToken/Spark-X2.6-4B")


if __name__ == "__main__":
    unittest.main()
