"""Tests for the minimal Spark-X2.6-4B health chatbot (stdlib unittest only)."""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_starts_with_system_prompt(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)

    def test_question_is_last_user_message(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})

    def test_history_is_kept_between_system_and_question(self):
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        messages = hc.build_messages("next", history=history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1:3], history)

    def test_system_prompt_forbids_diagnosis_and_prescribing(self):
        self.assertIn("Never diagnose", hc.SYSTEM_PROMPT)
        self.assertIn("never prescribe", hc.SYSTEM_PROMPT)
        self.assertIn("not medical advice", hc.SYSTEM_PROMPT)


class BuildPayloadTest(unittest.TestCase):
    def test_uses_model_id_and_token_budget(self):
        payload = hc.build_payload(hc.build_messages("q"))
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])

    def test_token_budget_can_be_overridden(self):
        payload = hc.build_payload(hc.build_messages("q"), max_new_tokens=64)
        self.assertEqual(payload["max_tokens"], 64)


class ExtractTextTest(unittest.TestCase):
    def test_returns_stripped_content(self):
        response = {"choices": [{"message": {"content": "  Drink water.  "}}]}
        self.assertEqual(hc.extract_text(response), "Drink water.")

    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            hc.extract_text({"choices": []})

    def test_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            hc.extract_text({"choices": [{"message": {"content": "   "}}]})


class EmergencyBannerTest(unittest.TestCase):
    def test_banner_added_for_emergency_question(self):
        answer = hc.add_emergency_banner("Rest and see a doctor.", "I have chest pain")
        self.assertTrue(answer.startswith(hc.EMERGENCY_BANNER))

    def test_banner_not_duplicated(self):
        text = hc.EMERGENCY_BANNER + "\n\nRest."
        self.assertEqual(hc.add_emergency_banner(text, "chest pain"), text)

    def test_no_banner_for_non_emergency_question(self):
        answer = "Rest and see a doctor."
        self.assertEqual(hc.add_emergency_banner(answer, "mild rash"), answer)


class AnswerQuestionTest(unittest.TestCase):
    def _response(self, content):
        return {"model": hc.MODEL_ID, "choices": [{"message": {"content": content}}]}

    def test_calls_endpoint_with_bounded_payload(self):
        with mock.patch.object(hc, "request_answer", return_value=self._response("See a clinician.")) as patched:
            result = hc.answer_question("I have a fever")
        payload = patched.call_args[0][0]
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["messages"][-1]["content"], "I have a fever")
        self.assertEqual(result["answer"], "See a clinician.")
        self.assertEqual(result["model"], hc.MODEL_ID)

    def test_emergency_banner_applied_to_model_answer(self):
        with mock.patch.object(hc, "request_answer", return_value=self._response("Call someone.")):
            result = hc.answer_question("I cannot breathe")
        self.assertTrue(result["answer"].startswith(hc.EMERGENCY_BANNER))

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            hc.answer_question("   ")


class MainTest(unittest.TestCase):
    def test_dry_run_prints_payload_without_network(self):
        buffer = io.StringIO()
        with mock.patch.object(hc, "request_answer", side_effect=AssertionError("network used")):
            with redirect_stdout(buffer):
                code = hc.main(["--dry-run"])
        printed = json.loads(buffer.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(printed["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(printed["messages"][-1]["content"], hc.DEMO_QUESTION)

    def test_returns_error_when_endpoint_unreachable(self):
        buffer = io.StringIO()
        with mock.patch.object(hc, "request_answer", side_effect=OSError("connection refused")):
            with redirect_stdout(buffer):
                code = hc.main(["-q", "hello"])
        self.assertEqual(code, 1)
        self.assertIn("Could not reach", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
