'''Tests for the minimal Spark-X2.6-4B health chatbot.

Run from the repository root:
    python src/test_health_chatbot.py
'''

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_starts_with_system_prompt(self):
        messages = hc.build_messages('What is a fever?')
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[0]['content'], hc.SYSTEM_PROMPT)

    def test_last_message_is_the_question(self):
        messages = hc.build_messages('  What is a fever?  ')
        self.assertEqual(messages[-1], {'role': 'user', 'content': 'What is a fever?'})

    def test_history_is_kept_in_order(self):
        history = [
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': 'hi'},
            {'role': 'tool', 'content': 'ignored'},
        ]
        messages = hc.build_messages('next', history)
        self.assertEqual(
            [m['role'] for m in messages], ['system', 'user', 'assistant', 'user']
        )


class PromptTest(unittest.TestCase):
    def test_fallback_prompt_contains_all_turns(self):
        messages = hc.build_messages('Is this urgent?')
        prompt = hc.build_prompt(messages)
        self.assertIn(hc.SYSTEM_PROMPT, prompt)
        self.assertIn('Is this urgent?', prompt)
        self.assertTrue(prompt.rstrip().endswith('<|assistant|>'))

    def test_uses_chat_template_when_available(self):
        class FakeTokenizer:
            chat_template = 'template'

            def apply_chat_template(
                self, messages, tokenize=False, add_generation_prompt=True
            ):
                assert tokenize is False and add_generation_prompt is True
                return 'TEMPLATED:' + messages[-1]['content']

        prompt = hc.build_prompt(hc.build_messages('question'), FakeTokenizer())
        self.assertEqual(prompt, 'TEMPLATED:question')


class PostprocessTest(unittest.TestCase):
    def test_disclaimer_added(self):
        out = hc.postprocess('Drink water and rest.')
        self.assertIn('Drink water and rest.', out)
        self.assertIn(hc.DISCLAIMER, out)

    def test_disclaimer_not_duplicated(self):
        once = hc.postprocess('Rest.')
        self.assertEqual(hc.postprocess(once), once)

    def test_empty_output_is_replaced(self):
        out = hc.postprocess('   ')
        self.assertIn(hc.DISCLAIMER, out)
        self.assertNotEqual(out.strip(), hc.DISCLAIMER)


class GenerateReplyTest(unittest.TestCase):
    def test_prompt_is_forwarded_and_disclaimer_appended(self):
        seen = {}

        def fake_generate(prompt):
            seen['prompt'] = prompt
            return 'Rest and monitor your temperature.'

        reply = hc.generate_reply(hc.DEMO_QUESTION, generate_fn=fake_generate)
        self.assertIn(hc.DEMO_QUESTION, seen['prompt'])
        self.assertIn(hc.SYSTEM_PROMPT, seen['prompt'])
        self.assertIn('Rest and monitor your temperature.', reply)
        self.assertIn(hc.DISCLAIMER, reply)

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.generate_reply('   ', generate_fn=lambda prompt: 'x')

    def test_system_prompt_forbids_diagnosis(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        self.assertIn('not diagnose', lowered)
        self.assertIn('clinician', lowered)


if __name__ == '__main__':
    unittest.main()
