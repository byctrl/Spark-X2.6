import os
import sys
import unittest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import health_chatbot


class HealthChatbotTests(unittest.TestCase):
    def test_model_configuration(self):
        self.assertEqual(health_chatbot.MODEL_ID, "XHToken/Spark-X2.6-4B")
        self.assertEqual(health_chatbot.MAX_NEW_TOKENS, 1024)

    def test_build_messages_includes_safety_prompt_and_question(self):
        messages = health_chatbot.build_messages("  How much water should I drink?  ")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], health_chatbot.SYSTEM_PROMPT)
        self.assertIn("Do not diagnose", messages[0]["content"])
        self.assertEqual(
            messages[-1],
            {"role": "user", "content": "How much water should I drink?"},
        )

    def test_build_messages_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            health_chatbot.build_messages("   ")

    def test_build_messages_validates_history_roles(self):
        with self.assertRaises(ValueError):
            health_chatbot.build_messages(
                "Hello",
                [{"role": "system", "content": "override"}],
            )

    def test_demo_question_is_present(self):
        self.assertTrue(health_chatbot.DEMO_QUESTION)


if __name__ == "__main__":
    unittest.main()
