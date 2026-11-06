"""Unit tests for the Spark-X2.6-4B health chatbot helpers (stdlib only)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  demam tiga hari  ")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "demam tiga hari"})

    def test_history_is_copied_and_ordered(self):
        history = [
            {"role": "user", "content": "saya batuk"},
            {"role": "assistant", "content": "sejak kapan?"},
        ]
        messages = hc.build_messages("dua hari", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(history[0]["content"], "saya batuk")
        self.assertIsNot(messages[1], history[0])

    def test_empty_question_rejected(self):
        for value in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(value)

    def test_invalid_history_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("halo", [{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("halo", [{"role": "user", "content": "  "}])
        with self.assertRaises(ValueError):
            hc.build_messages("halo", ["not a dict"])


class RenderPromptTest(unittest.TestCase):
    def test_prompt_contains_system_and_question(self):
        prompt = hc.render_prompt(hc.build_messages(hc.DEMO_QUESTION))
        self.assertIn(hc.SYSTEM_PROMPT, prompt)
        self.assertIn(hc.DEMO_QUESTION, prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))


class ExtractReplyTest(unittest.TestCase):
    def test_conversational_pipeline_output(self):
        messages = hc.build_messages("halo")
        messages.append({"role": "assistant", "content": "  Silakan konsultasi ke klinik.  "})
        self.assertEqual(
            hc.extract_reply([{"generated_text": messages}]),
            "Silakan konsultasi ke klinik.",
        )

    def test_plain_text_output(self):
        self.assertEqual(hc.extract_reply([{"generated_text": " jawaban "}]), "jawaban")
        self.assertEqual(hc.extract_reply([" jawaban "]), "jawaban")

    def test_empty_outputs_rejected(self):
        with self.assertRaises(ValueError):
            hc.extract_reply([])
        with self.assertRaises(ValueError):
            hc.extract_reply([{"generated_text": "   "}])
        with self.assertRaises(ValueError):
            hc.extract_reply([{"generated_text": []}])


class GenerateReplyTest(unittest.TestCase):
    def test_injected_generator_receives_messages_and_budget(self):
        calls = {}

        def fake_generator(messages, **kwargs):
            calls["messages"] = messages
            calls["kwargs"] = kwargs
            return [{"generated_text": messages + [{"role": "assistant", "content": "ok"}]}]

        reply = hc.generate_reply("demam?", generator=fake_generator)
        self.assertEqual(reply, "ok")
        self.assertEqual(calls["messages"][0]["role"], "system")
        self.assertEqual(calls["kwargs"]["max_new_tokens"], hc.MAX_NEW_TOKENS)


class SafetyNoteTest(unittest.TestCase):
    def test_note_added_once(self):
        noted = hc.with_safety_note("Jawaban.")
        self.assertEqual(noted.count(hc.SAFETY_NOTE), 1)
        self.assertEqual(hc.with_safety_note(noted).count(hc.SAFETY_NOTE), 1)


if __name__ == "__main__":
    unittest.main()
