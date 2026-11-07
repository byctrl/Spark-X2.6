"""Small stdlib-only tests for src/health_chatbot.py (no model download)."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import health_chatbot as hc  # noqa: E402


class _FakeTokenizer:
    def __init__(self, prompt_ids=(1, 2, 3)):
        self.prompt_ids = list(prompt_ids)
        self.last_prompt = None

    def __call__(self, prompt, return_tensors=None):
        self.last_prompt = prompt
        return {"input_ids": [list(self.prompt_ids)]}

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(token) for token in tokens)


class _FakeModel:
    device = None

    def __init__(self, new_tokens=(101, 102)):
        self.new_tokens = list(new_tokens)
        self.kwargs = None

    def generate(self, input_ids, max_new_tokens=None, do_sample=None):
        self.kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        return [list(input_ids[0]) + self.new_tokens]


class ConstantsTest(unittest.TestCase):
    def test_model_id(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")

    def test_max_new_tokens_is_bounded(self):
        self.assertIsInstance(hc.MAX_NEW_TOKENS, int)
        self.assertGreater(hc.MAX_NEW_TOKENS, 0)
        self.assertLessEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_system_prompt_keeps_safety_boundaries(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("not diagnose", prompt)
        self.assertIn("clinician", prompt)
        self.assertIn("emergency", prompt)

    def test_demo_question_is_present(self):
        self.assertTrue(hc.DEMO_QUESTION.strip())


class BuildMessagesTest(unittest.TestCase):
    def test_structure(self):
        messages = hc.build_messages("  I have a cough  ")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "I have a cough")

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_rejects_non_string(self):
        with self.assertRaises(ValueError):
            hc.build_messages(None)


class FormatPromptTest(unittest.TestCase):
    def test_renders_roles_and_opens_assistant_turn(self):
        prompt = hc.format_prompt(hc.build_messages("hello"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertIn("hello", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>\n"))


class GenerateTest(unittest.TestCase):
    def test_only_new_tokens_are_decoded(self):
        tokenizer = _FakeTokenizer(prompt_ids=(1, 2, 3))
        model = _FakeModel(new_tokens=(101, 102))
        answer = hc.generate("hello", model=model, tokenizer=tokenizer)
        self.assertEqual(answer, "101 102")

    def test_prompt_uses_system_and_user_turns(self):
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        hc.generate("hello", model=model, tokenizer=tokenizer)
        self.assertIn(hc.SYSTEM_PROMPT, tokenizer.last_prompt)
        self.assertIn("hello", tokenizer.last_prompt)

    def test_generation_is_deterministic_and_bounded(self):
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        hc.generate("hello", model=model, tokenizer=tokenizer, max_new_tokens=7)
        self.assertEqual(model.kwargs["max_new_tokens"], 7)
        self.assertFalse(model.kwargs["do_sample"])


if __name__ == "__main__":
    unittest.main()
