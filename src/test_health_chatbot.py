"""Offline tests for the minimal Spark-X2.6-4B health chatbot demo."""

import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_comes_first(self):
        messages = hc.build_messages("What is a fever?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(len(messages), 2)

    def test_question_is_trimmed_and_last(self):
        messages = hc.build_messages("   Apa itu demam?  ")
        self.assertEqual(messages[-1], {"role": "user", "content": "Apa itu demam?"})

    def test_history_is_kept_in_order_and_sanitised(self):
        history = [
            {"role": "user", "content": "Halo"},
            {"role": "assistant", "content": "Halo, ada yang bisa saya bantu?"},
            {"role": "tool", "content": "ignored"},
            {"role": "assistant", "content": "   "},
            "not-a-dict",
        ]
        messages = hc.build_messages("Saya demam", history=history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(messages[-1]["content"], "Saya demam")

    def test_empty_question_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_system_prompt_forbids_diagnosis(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("not a diagnosis", prompt)


def _payload_test_placeholder():
    pass


class BuildPayloadTest(unittest.TestCase):
    def test_payload_uses_model_and_token_budget(self):
        payload = hc.build_payload(hc.build_messages("hi"))
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(payload["stream"])
        self.assertEqual(len(payload["messages"]), 2)


class ParseResponseTest(unittest.TestCase):
    def test_extracts_content(self):
        data = {
            "choices": [
                {"message": {"role": "assistant", "content": "  Istirahat dan minum air.  "}}
            ]
        }
        self.assertEqual(hc.parse_response(data), "Istirahat dan minum air.")

    def test_rejects_malformed_or_empty(self):
        bad_payloads = [
            {},
            {"choices": []},
            {"choices": [{}]},
            None,
            {"choices": [{"message": {"content": "   "}}]},
        ]
        for bad in bad_payloads:
            with self.assertRaises(ValueError):
                hc.parse_response(bad)


if __name__ == "__main__":
    unittest.main()
