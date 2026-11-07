"""Tests for the Spark-X2.6-4B health-information chatbot."""
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import health_chatbot as hc  # noqa: E402


class FakeTokenizer:
    def __init__(self):
        self.prompt = None

    def __call__(self, prompt, return_tensors="pt"):
        self.prompt = prompt
        return {"input_ids": [11, 12, 13]}

    def decode(self, tokens, skip_special_tokens=True):
        return "Rest and drink water. %s" % list(tokens)


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.seen_max_new_tokens = None

    def generate(self, input_ids, max_new_tokens):
        self.seen_max_new_tokens = max_new_tokens
        return [[11, 12, 13, 21, 22]]


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_is_first_and_question_is_last(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_system_prompt_forbids_diagnosis(self):
        system = hc.build_messages("hi")[0]["content"].lower()
        self.assertIn("do not diagnose", system)
        self.assertIn("clinician", system)

    def test_history_is_reused_without_mutation(self):
        history = [{"role": "user", "content": "I have a cough."}]
        messages = hc.build_messages("Since when?", history)
        self.assertEqual(len(history), 1)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "user"])

    def test_invalid_history_entries_are_skipped(self):
        history = [{"role": "tool", "content": "x"}, {"role": "assistant", "content": " "}]
        messages = hc.build_messages("next", history)
        self.assertEqual(len(messages), 2)

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class SafetyTest(unittest.TestCase):
    def test_urgent_terms_detected(self):
        self.assertTrue(hc.needs_urgent_care("I have chest pain right now"))
        self.assertTrue(hc.needs_urgent_care("他突然胸痛"))
        self.assertFalse(hc.needs_urgent_care("I have a mild rash"))

    def test_reply_always_carries_disclaimer(self):
        reply = hc.decorate_reply("Rest and monitor.", "I have a mild rash")
        self.assertIn(hc.DISCLAIMER, reply)
        self.assertNotIn(hc.URGENT_NOTICE, reply)

    def test_urgent_question_adds_emergency_notice(self):
        reply = hc.decorate_reply("Call your doctor.", "difficulty breathing since morning")
        self.assertIn(hc.URGENT_NOTICE, reply)
        self.assertIn(hc.DISCLAIMER, reply)


class GenerationTest(unittest.TestCase):
    def test_generate_reply_uses_model_and_decodes_new_tokens(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()
        reply = hc.generate_reply(
            "What is a fever?", model=model, tokenizer=tokenizer
        )
        self.assertEqual(model.seen_max_new_tokens, hc.MAX_NEW_TOKENS)
        self.assertIn("Rest and drink water. [21, 22]", reply)
        self.assertIn(hc.DISCLAIMER, reply)
        self.assertIn("What is a fever?", tokenizer.prompt)

    def test_generate_reply_requires_matching_model_and_tokenizer(self):
        with self.assertRaises(ValueError):
            hc.generate_reply("hi", model=FakeModel())
        with self.assertRaises(ValueError):
            hc.generate_reply("hi", tokenizer=FakeTokenizer())

    def test_fallback_render_contains_roles(self):
        prompt = hc.fallback_render(hc.build_messages("hello"))
        self.assertIn("System:", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_model_id_points_to_spark(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertTrue(hc.DEMO_QUESTION.strip())


if __name__ == "__main__":
    unittest.main()
