"""Tests for the Spark-X2.6-4B health chatbot demo (no model download needed)."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import health_chatbot as chatbot  # noqa: E402


class BuildMessagesTest(unittest.TestCase):
    def test_model_id_and_budget(self):
        self.assertEqual(chatbot.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(chatbot.MAX_NEW_TOKENS, 1024)

    def test_demo_question_is_set(self):
        self.assertTrue(chatbot.DEMO_QUESTION.strip())

    def test_messages_have_system_and_user(self):
        messages = chatbot.build_messages("  How can I avoid dehydration?  ")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "How can I avoid dehydration?")

    def test_system_prompt_blocks_diagnosis(self):
        system = chatbot.build_messages("test")[0]["content"]
        self.assertIn("Do not diagnose", system)
        self.assertIn("consult a qualified health worker", system)

    def test_language_is_carried_into_system_prompt(self):
        system = chatbot.build_messages("test", language="id")[0]["content"]
        self.assertIn("id", system)

    def test_empty_question_is_rejected(self):
        for value in ("", "   ", None):
            with self.assertRaises(ValueError):
                chatbot.build_messages(value)


class ExtractReplyTest(unittest.TestCase):
    def test_chat_style_output(self):
        output = [
            {
                "generated_text": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "  Drink water.  "},
                ]
            }
        ]
        self.assertEqual(chatbot.extract_reply(output), "Drink water.")

    def test_plain_text_output(self):
        self.assertEqual(chatbot.extract_reply([{"generated_text": " hi "}]), "hi")

    def test_missing_assistant_message(self):
        output = [{"generated_text": [{"role": "user", "content": "hi"}]}]
        self.assertEqual(chatbot.extract_reply(output), "")

    def test_empty_output(self):
        self.assertEqual(chatbot.extract_reply([]), "")


class AnswerTest(unittest.TestCase):
    def test_answer_uses_injected_pipeline(self):
        seen = {}

        def fake_pipeline(messages, **kwargs):
            seen["messages"] = messages
            seen["kwargs"] = kwargs
            return [
                {
                    "generated_text": messages
                    + [{"role": "assistant", "content": " Talk to a clinician. "}]
                }
            ]

        reply = chatbot.answer("How much water is enough?", pipe=fake_pipeline)
        self.assertEqual(reply, "Talk to a clinician.")
        self.assertEqual(seen["kwargs"]["max_new_tokens"], chatbot.MAX_NEW_TOKENS)
        self.assertEqual(seen["messages"][1]["content"], "How much water is enough?")

    def test_empty_reply_raises(self):
        with self.assertRaises(RuntimeError):
            chatbot.answer("anything", pipe=lambda messages, **kwargs: [])


if __name__ == "__main__":
    unittest.main()
