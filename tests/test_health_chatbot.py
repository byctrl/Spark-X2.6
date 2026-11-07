import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"),
)

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("  I have a fever  ")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[1], {"role": "user", "content": "I have a fever"})

    def test_empty_question_rejected(self):
        for value in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(value)

    def test_prompt_forbids_diagnosis_and_prescription(self):
        self.assertIn("Never diagnose", hc.SYSTEM_PROMPT)
        self.assertIn("prescribe", hc.SYSTEM_PROMPT)


class DisclaimerTest(unittest.TestCase):
    def test_disclaimer_appended_once(self):
        once = hc.with_disclaimer("Rest and drink fluids.")
        self.assertTrue(once.startswith("Rest and drink fluids."))
        self.assertIn(hc.DISCLAIMER, once)
        self.assertEqual(hc.with_disclaimer(once), once)

    def test_blank_answer_returns_disclaimer(self):
        self.assertEqual(hc.with_disclaimer(""), hc.DISCLAIMER)
        self.assertEqual(hc.with_disclaimer(None), hc.DISCLAIMER)


class GenerateReplyTest(unittest.TestCase):
    def test_generator_receives_messages_and_budget(self):
        seen = {}

        def fake_generator(messages, max_new_tokens=hc.MAX_NEW_TOKENS):
            seen["messages"] = messages
            seen["max_new_tokens"] = max_new_tokens
            return "A clinician may check your temperature and breathing."

        reply = hc.generate_reply(
            hc.DEMO_QUESTION, generator=fake_generator, max_new_tokens=64
        )
        self.assertEqual(seen["max_new_tokens"], 64)
        self.assertEqual(seen["messages"][1]["content"], hc.DEMO_QUESTION)
        self.assertIn(hc.DISCLAIMER, reply)

    def test_empty_model_output_still_guarded(self):
        reply = hc.generate_reply(
            hc.DEMO_QUESTION, generator=lambda messages, max_new_tokens=0: "  "
        )
        self.assertEqual(reply, hc.DISCLAIMER)

    def test_model_id_pinned(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")


if __name__ == "__main__":
    unittest.main()
