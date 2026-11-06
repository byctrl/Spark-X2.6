"""Offline tests for the health chatbot demo (no model download required)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import health_chatbot as hc


class DummyTokenizer:
    """Stand-in tokenizer without a chat template."""


class ChatTemplateTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [m["role"] + ": " + m["content"] for m in messages]
        return "\n".join(parts) + ("\nassistant:" if add_generation_prompt else "")


def test_model_settings():
    assert hc.MODEL_ID == "XHToken/Spark-X2.6-4B"
    assert hc.MAX_NEW_TOKENS == 1024


def test_build_messages_shape():
    messages = hc.build_messages("  how do I treat a fever?  ")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "how do I treat a fever?"


def test_system_prompt_forbids_diagnosis():
    prompt = hc.SYSTEM_PROMPT.lower()
    assert "do not diagnose" in prompt
    assert "qualified clinician" in prompt


def test_build_messages_rejects_empty():
    for bad in ("", "   \n"):
        try:
            hc.build_messages(bad)
        except ValueError:
            continue
        raise AssertionError("empty question should raise ValueError")


def test_build_prompt_uses_chat_template_when_present():
    prompt = hc.build_prompt(ChatTemplateTokenizer(), "hello")
    assert prompt.startswith("system: ")
    assert prompt.endswith("assistant:")
    assert "hello" in prompt


def test_build_prompt_falls_back_without_template():
    prompt = hc.build_prompt(DummyTokenizer(), "hello")
    assert "SYSTEM: " in prompt
    assert "USER: hello" in prompt
    assert prompt.endswith("ASSISTANT:")


def test_format_output_appends_footer():
    assert hc.format_output("Drink water and rest.").endswith(hc.SAFETY_FOOTER)


def test_format_output_does_not_duplicate_footer():
    text = "Rest, and confirm with a qualified clinician."
    assert hc.format_output(text) == text


def test_format_output_handles_empty_reply():
    out = hc.format_output("")
    assert "could not produce an answer" in out
    assert hc.SAFETY_FOOTER in out


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print("PASS " + name)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("FAIL " + name + ": " + repr(exc))
    raise SystemExit(1 if failures else 0)
