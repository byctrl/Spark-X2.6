"""Offline unit tests for src/health_chatbot.py (no network, no model download)."""

import unittest

import health_chatbot as hc


class FakeTransport(object):
    def __init__(self, response=None, error=None):
        self.response = response or {
            "choices": [{"message": {"role": "assistant", "content": "  Drink water.  "}}]
        }
        self.error = error
        self.calls = []

    def __call__(self, payload, base_url):
        self.calls.append((payload, base_url))
        if self.error:
            raise self.error
        return self.response


class BuildMessagesTest(unittest.TestCase):
    def test_system_prompt_first_and_question_last(self):
        messages = hc.build_messages("  demam ringan  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], hc.SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "demam ringan"})

    def test_system_prompt_forbids_diagnosis(self):
        self.assertIn("Never diagnose", hc.SYSTEM_PROMPT)
        self.assertIn("not medical advice", hc.SYSTEM_PROMPT)

    def test_history_is_kept_between_system_prompt_and_question(self):
        history = [
            {"role": "user", "content": "sakit kepala"},
            {"role": "assistant", "content": "Sejak kapan?"},
        ]
        messages = hc.build_messages("tiga hari", history=history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1], {"role": "user", "content": "sakit kepala"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "Sejak kapan?"})
        self.assertEqual(messages[3]["content"], "tiga hari")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("   ")
        with self.assertRaises(ValueError):
            hc.build_messages(None)

    def test_invalid_history_is_rejected(self):
        with self.assertRaises(ValueError):
            hc.build_messages("halo", history=[{"role": "system", "content": "x"}])
        with self.assertRaises(ValueError):
            hc.build_messages("halo", history=["bos"])


class BuildPayloadTest(unittest.TestCase):
    def test_payload_targets_spark_model(self):
        payload = hc.build_payload("batuk")
        self.assertEqual(payload["model"], "XHToken/Spark-X2.6-4B")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["messages"][0]["role"], "system")


class ExtractTextTest(unittest.TestCase):
    def test_strips_content(self):
        response = {"choices": [{"message": {"content": "  jawaban  "}}]}
        self.assertEqual(hc.extract_text(response), "jawaban")

    def test_rejects_broken_or_empty_response(self):
        with self.assertRaises(ValueError):
            hc.extract_text({})
        with self.assertRaises(ValueError):
            hc.extract_text({"choices": [{"message": {"content": "   "}}]})


class HealthChatTest(unittest.TestCase):
    def test_uses_injected_transport_and_returns_answer(self):
        transport = FakeTransport()
        answer = hc.health_chat(
            "batuk kering dan demam",
            base_url="http://127.0.0.1:30000/v1",
            transport=transport,
        )
        self.assertEqual(answer, "Drink water.")
        self.assertEqual(transport.calls[0][1], "http://127.0.0.1:30000/v1")
        payload = transport.calls[0][0]
        self.assertEqual(payload["model"], hc.MODEL_ID)
        self.assertEqual(payload["messages"][-1]["content"], "batuk kering dan demam")

    def test_transport_errors_are_propagated(self):
        transport = FakeTransport(error=RuntimeError("endpoint down"))
        with self.assertRaises(RuntimeError):
            hc.health_chat("demam", transport=transport)

    def test_render_answer_includes_safety_notice(self):
        rendered = hc.render_answer("Minum air.")
        self.assertIn("Minum air.", rendered)
        self.assertIn(hc.DISCLAIMER, rendered)


if __name__ == "__main__":
    unittest.main()
