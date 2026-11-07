"""Small, dependency-free tests for the Spark-X2.6-4B health chatbot."""

import unittest

import health_chatbot as hc


class BuildMessagesTest(unittest.TestCase):
    def test_includes_system_prompt_and_question(self):
        messages = hc.build_messages("  What is a fever?  ")
        self.assertEqual(messages[0], {"role": "system", "content": hc.SYSTEM_PROMPT})
        self.assertEqual(messages[-1], {"role": "user", "content": "What is a fever?"})
        self.assertEqual(len(messages), 2)

    def test_rejects_empty_question(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(ValueError):
                hc.build_messages(bad)

    def test_history_is_filtered_and_ordered(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "ignore me"},
            {"role": "assistant", "content": "hi there"},
            {"role": "assistant", "content": "   "},
        ]
        messages = hc.build_messages("next question", history)
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"],
        )

    def test_system_prompt_forbids_diagnosis(self):
        lowered = hc.SYSTEM_PROMPT.lower()
        self.assertIn("never diagnose", lowered)
        self.assertIn("clinician", lowered)
        self.assertIn("urgent care", lowered)

    def test_model_constants(self):
        self.assertEqual(hc.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(hc.MAX_NEW_TOKENS, 1024)

    def test_demo_question_is_a_question(self):
        self.assertTrue(hc.DEMO_QUESTION.endswith("?"))


class AskTest(unittest.TestCase):
    def _bot(self, generated_text):
        bot = hc.HealthChatbot()
        bot._pipe = lambda messages, **kwargs: [{"generated_text": generated_text}]
        return bot

    def test_chat_style_output_is_used(self):
        bot = self._bot(
            [
                {"role": "system", "content": hc.SYSTEM_PROMPT},
                {"role": "user", "content": "I have a cough"},
                {"role": "assistant", "content": "Please see a clinician."},
            ]
        )
        reply = bot.ask("I have a cough")
        self.assertIn("Please see a clinician.", reply)
        self.assertIn(hc.DISCLAIMER, reply)
        self.assertNotIn(hc.SYSTEM_PROMPT, reply)

    def test_plain_text_output_is_used(self):
        bot = self._bot("Drink water and rest.\n")
        reply = bot.ask("I have a cough")
        self.assertTrue(reply.startswith("Drink water and rest."))
        self.assertIn(hc.DISCLAIMER, reply)

    def test_empty_output_falls_back_to_clinician(self):
        reply = self._bot("   ").ask("I have a cough")
        self.assertIn("ask a clinician", reply)
        self.assertIn(hc.DISCLAIMER, reply)

    def test_pipeline_receives_safety_settings(self):
        seen = {}
        bot = hc.HealthChatbot(max_new_tokens=64)

        def fake_pipe(messages, **kwargs):
            seen["messages"] = messages
            seen.update(kwargs)
            return [{"generated_text": "ok"}]

        bot._pipe = fake_pipe
        bot.ask("I have a cough")
        self.assertEqual(seen["max_new_tokens"], 64)
        self.assertFalse(seen["do_sample"])
        self.assertEqual(seen["messages"][0]["role"], "system")

    def test_invalid_question_raises_before_model_call(self):
        bot = hc.HealthChatbot()
        bot._pipe = lambda messages, **kwargs: self.fail("model must not be called")
        with self.assertRaises(ValueError):
            bot.ask("  ")


if __name__ == "__main__":
    unittest.main()
