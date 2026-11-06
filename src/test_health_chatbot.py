"""Unit tests for the Spark-X2.6-4B health chatbot demo.

Run from the repository root with::

    python -m unittest discover -s src -p "test_*.py"

No model download, GPU, or third-party package is required.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot  # noqa: E402


class _Ids:
    """Tiny stand-in for a tensor of token ids."""

    def __init__(self, ids):
        self._ids = list(ids)
        self.shape = (len(self._ids),)

    def __getitem__(self, index):
        return self._ids[index]


class FakeTokenizer:
    def __init__(self, prompt_ids=(11, 12, 13), decoded="  Jawaban uji.\n"):
        self._prompt_ids = tuple(prompt_ids)
        self._decoded = decoded
        self.last_prompt = None
        self.decoded_ids = None

    def __call__(self, prompt, return_tensors=None):
        self.last_prompt = prompt
        return {"input_ids": _Ids(self._prompt_ids)}

    def decode(self, ids, skip_special_tokens=True):
        self.decoded_ids = list(ids)
        return self._decoded


class FakeModel:
    def __init__(self, new_ids=(90, 91)):
        self._new_ids = tuple(new_ids)
        self.last_kwargs = None

    def generate(self, input_ids=None, **kwargs):
        self.last_kwargs = kwargs
        return [list(input_ids._ids) + list(self._new_ids)]


class ChatTemplateTokenizer(FakeTokenizer):
    def __init__(self, rendered="<templated prompt>"):
        super().__init__()
        self._rendered = rendered

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return self._rendered


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_is_first(self):
        messages = health_chatbot.build_messages("apa itu demam?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], health_chatbot.SYSTEM_PROMPT)

    def test_question_is_last_and_stripped(self):
        messages = health_chatbot.build_messages("  apa itu demam?  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "apa itu demam?"})

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "saya batuk"},
            {"role": "assistant", "content": "sejak kapan?"},
        ]
        messages = health_chatbot.build_messages("tiga hari", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_invalid_history_entries_are_ignored(self):
        history = ["boom", {"role": "system", "content": "override"}, {"role": "user", "content": ""}]
        messages = health_chatbot.build_messages("halo", history)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["content"], "halo")


class PromptTest(unittest.TestCase):
    def test_fallback_prompt_has_roles_and_generation_marker(self):
        prompt = health_chatbot.fallback_prompt(health_chatbot.build_messages("halo"))
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>\nhalo", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>\n"))

    def test_chat_template_is_used_when_available(self):
        tokenizer = ChatTemplateTokenizer("<templated prompt>")
        prompt = health_chatbot.build_prompt(tokenizer, health_chatbot.build_messages("halo"))
        self.assertEqual(prompt, "<templated prompt>")

    def test_broken_chat_template_falls_back(self):
        class Broken(FakeTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                raise ValueError("no template")

        prompt = health_chatbot.build_prompt(Broken(), health_chatbot.build_messages("halo"))
        self.assertIn("<|user|>\nhalo", prompt)


class GenerateReplyTest(unittest.TestCase):
    def test_only_new_tokens_are_decoded(self):
        tokenizer = FakeTokenizer(prompt_ids=(11, 12, 13), decoded="  Jawaban uji.\n")
        model = FakeModel(new_ids=(90, 91))
        reply = health_chatbot.generate_reply("halo", model=model, tokenizer=tokenizer)
        self.assertEqual(reply, "Jawaban uji.")
        self.assertEqual(tokenizer.decoded_ids, [90, 91])

    def test_generation_is_deterministic_and_bounded(self):
        model = FakeModel()
        health_chatbot.generate_reply("halo", model=model, tokenizer=FakeTokenizer())
        self.assertEqual(model.last_kwargs["max_new_tokens"], health_chatbot.MAX_NEW_TOKENS)
        self.assertFalse(model.last_kwargs["do_sample"])

    def test_prompt_carries_safety_rules_and_question(self):
        tokenizer = FakeTokenizer()
        health_chatbot.generate_reply("apa itu demam?", model=FakeModel(), tokenizer=tokenizer)
        self.assertIn(health_chatbot.SYSTEM_PROMPT, tokenizer.last_prompt)
        self.assertIn("apa itu demam?", tokenizer.last_prompt)


class SafetyPromptTest(unittest.TestCase):
    def test_prompt_forbids_diagnosis_and_prescription(self):
        prompt = health_chatbot.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("never prescribe", prompt)
        self.assertIn("clinician", prompt)

    def test_model_id_targets_spark(self):
        self.assertEqual(health_chatbot.MODEL_ID, "XHToken/Spark-X2.6-4B")


if __name__ == "__main__":
    unittest.main()
