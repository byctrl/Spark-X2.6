"""Offline tests for src/health_chatbot.py (no model, no network)."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})
        self.assertEqual(len(messages), 2)

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")

    def test_history_is_filtered_and_kept_in_order(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": ""},
            "not-a-dict",
        ]
        messages = hc.build_messages("next", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])


class PayloadTest(unittest.TestCase):
    def test_payload_targets_spark_x2_6(self):
        payload = hc.build_payload(hc.build_messages("q"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])

    def test_max_tokens_override(self):
        payload = hc.build_payload(hc.build_messages("q"), max_new_tokens=64)
        self.assertEqual(payload["max_tokens"], 64)


class ExtractAnswerTest(unittest.TestCase):
    def test_reads_chat_completion(self):
        body = {"choices": [{"message": {"role": "assistant", "content": " See a nurse. "}}]}
        self.assertEqual(hc.extract_answer(body), "See a nurse.")

    def test_bad_body_raises(self):
        for body in ({}, {"choices": []}, {"choices": [{}]}, None):
            with self.assertRaises(RuntimeError):
                hc.extract_answer(body)


class SafetyTest(unittest.TestCase):
    def test_emergency_cue_is_flagged(self):
        self.assertEqual(hc.emergency_notice("I have Chest Pain right now"), hc.EMERGENCY_NOTICE)
        self.assertIn(hc.EMERGENCY_NOTICE, hc.answer_question("胸痛怎么办"))

    def test_ordinary_question_passes_through(self):
        self.assertEqual(hc.emergency_notice("what is a fever?"), "")

    def test_disclaimer_added_once(self):
        once = hc.with_disclaimer("Drink water.")
        self.assertIn(hc.DISCLAIMER, once)
        self.assertEqual(hc.with_disclaimer(once), once)
        self.assertEqual(hc.with_disclaimer(""), hc.DISCLAIMER)

    def test_answer_question_wraps_generator(self):
        answer = hc.answer_question("what is a fever?", generator=lambda q, h: "A fever is a high temperature.")
        self.assertIn("A fever is a high temperature.", answer)
        self.assertIn(hc.DISCLAIMER, answer)

    def test_emergency_skips_model_call(self):
        def boom(question, history):
            raise AssertionError("model must not be called for emergencies")

        self.assertEqual(hc.answer_question("cannot breathe", generator=boom), hc.EMERGENCY_NOTICE)


if __name__ == "__main__":
    unittest.main()
