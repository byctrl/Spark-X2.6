'''Offline tests for src/health_chatbot.py (no model download needed).'''

import os
import sys
import unittest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
)

import health_chatbot as hb  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hb.build_messages('fever for two days')
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[0]['content'], hb.SYSTEM_PROMPT)
        self.assertEqual(
            messages[-1], {'role': 'user', 'content': 'fever for two days'}
        )

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hb.build_messages('   ')

    def test_history_kept_between_system_and_question(self):
        history = [
            {'role': 'user', 'content': 'I have a cough'},
            {'role': 'assistant', 'content': 'Since when?'},
            {'role': 'system', 'content': 'ignored'},
            {'role': 'user', 'content': '   '},
        ]
        messages = hb.build_messages('three days', history)
        self.assertEqual(
            [m['role'] for m in messages],
            ['system', 'user', 'assistant', 'user'],
        )

    def test_system_prompt_is_safety_bounded(self):
        lowered = hb.SYSTEM_PROMPT.lower()
        self.assertIn('do not diagnose', lowered)
        self.assertIn('prescribe', lowered)
        self.assertIn('clinician', lowered)
        self.assertIn('identifying data', lowered)


class ReplyHandlingTest(unittest.TestCase):
    def test_payload_uses_pinned_model_and_budget(self):
        payload = hb.request_payload(hb.build_messages('hello'))
        self.assertEqual(payload['model'], 'XHToken/Spark-X2.6-4B')
        self.assertEqual(payload['max_tokens'], 1024)

    def test_parse_reply_reads_content(self):
        payload = {'choices': [{'message': {'content': '  Rest and fluids.  '}}]}
        self.assertEqual(hb.parse_reply(payload), 'Rest and fluids.')

    def test_parse_reply_rejects_bad_shapes(self):
        for payload in ({}, {'choices': []}, {'choices': [{'message': {}}]}, None):
            with self.assertRaises(ValueError):
                hb.parse_reply(payload)

    def test_disclaimer_added_once(self):
        once = hb.with_disclaimer('Drink water.')
        self.assertTrue(once.endswith(hb.DISCLAIMER))
        self.assertEqual(hb.with_disclaimer(once), once)


if __name__ == '__main__':
    unittest.main()
