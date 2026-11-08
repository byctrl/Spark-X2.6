"""Stdlib-only tests for src/health_chatbot.py.

Run from the repository root:
    python -m unittest discover -s src -p "test_*.py" -v

No model server is needed: generation is tested through an injected
fake HTTP poster, so the prompt/payload contract is verified offline.
"""

import unittest

import health_chatbot as hc


def _fake_post(content="General guidance."):
    calls = {}

    def post(url, payload, api_key):
        calls["url"] = url
        calls["payload"] = payload
        calls["api_key"] = api_key
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    return post, calls


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("What is oral rehydration solution?")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("oral rehydration", messages[-1]["content"])
        self.assertEqual(len(messages), 2)

    def test_history_is_kept_in_order(self):
        history = [
            {"role": "user", "content": "I have a mild cough."},
            {"role": "assistant", "content": "How long has it lasted?"},
        ]
        messages = hc.build_messages("About ten days.", history=history)
        self.assertEqual(
            [m["role"] for m in messages], ["system", "user", "assistant", "user"]
        )

    def test_invalid_history_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("hello", history=[{"role": "tool", "content": "x"}])

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")


class PayloadAndParsingTest(unittest.TestCase):
    def test_payload_targets_spark_model(self):
        payload = hc.build_payload("hello")
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)
        self.assertEqual(payload["max_tokens"], hc.MAX_NEW_TOKENS)
        self.assertFalse(payload["stream"])

    def test_system_prompt_blocks_diagnosis(self):
        prompt = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", prompt)
        self.assertIn("never prescribe", prompt)
        self.assertIn("clinician", prompt)

    def test_parse_openai_chat_response(self):
        data = {"choices": [{"message": {"role": "assistant", "content": "Drink fluids."}}]}
        self.assertEqual(hc.parse_response(data), "Drink fluids.")

    def test_parse_text_completion_response(self):
        self.assertEqual(hc.parse_response({"choices": [{"text": "Rest."}]}), "Rest.")

    def test_parse_unusable_response_raises(self):
        with self.assertRaises(RuntimeError):
            hc.parse_response({"choices": []})


class GenerateReplyTest(unittest.TestCase):
    def test_calls_endpoint_with_spark_payload(self):
        post, calls = _fake_post("Here is general information.")
        answer = hc.generate_reply(
            "What is oral rehydration solution?",
            http_post=post,
            base_url="http://127.0.0.1:30000/v1",
            api_key="EMPTY",
        )
        self.assertEqual(answer, "Here is general information.")
        self.assertEqual(calls["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(calls["payload"]["model"], hc.MODEL_ID)
        self.assertEqual(calls["payload"]["messages"][0]["role"], "system")
        self.assertEqual(calls["api_key"], "EMPTY")

    def test_empty_question_rejected_before_calling_model(self):
        post, calls = _fake_post()
        with self.assertRaises(ValueError):
            hc.generate_reply("", http_post=post)
        self.assertEqual(calls, {})

    def test_emergency_notice_is_prepended(self):
        post, _ = _fake_post("General guidance.")
        answer = hc.generate_reply("I have chest pain and cannot breathe", http_post=post)
        self.assertTrue(answer.startswith(hc.EMERGENCY_NOTICE))
        self.assertIn("General guidance.", answer)

    def test_emergency_notice_for_chinese_red_flag(self):
        post, _ = _fake_post("General guidance.")
        answer = hc.generate_reply("胸痛很严重，该怎么办？", http_post=post)
        self.assertTrue(answer.startswith(hc.EMERGENCY_NOTICE))

    def test_no_notice_for_ordinary_question(self):
        post, _ = _fake_post("General guidance.")
        answer = hc.generate_reply(
            "How much water should an adult drink daily?", http_post=post
        )
        self.assertEqual(answer, "General guidance.")


if __name__ == "__main__":
    unittest.main()
