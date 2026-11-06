"""Small offline tests for the Spark-X2.6-4B health-information chatbot."""

import health_chatbot
from health_chatbot import (
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    ask,
    build_messages,
    build_payload,
    parse_response,
)


def test_system_prompt_stays_safety_bounded():
    prompt = SYSTEM_PROMPT.lower()
    assert "do not diagnose" in prompt
    assert "emergency" in prompt
    assert MODEL_ID == "XHToken/Spark-X2.6-4B"


def test_build_messages_keeps_system_prompt_and_question():
    messages = build_messages("  I have a cough  ")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "I have a cough"}


def test_build_messages_rejects_empty_question():
    try:
        build_messages("   ")
    except ValueError:
        return
    raise AssertionError("empty questions must be rejected")


def test_build_payload_uses_model_and_token_limit():
    messages = build_messages("When should I see a doctor?")
    payload = build_payload(messages)
    assert payload["model"] == MODEL_ID
    assert payload["max_tokens"] == MAX_NEW_TOKENS
    assert payload["messages"] == messages
    assert 0.0 <= float(payload["temperature"]) <= 1.0


def test_parse_response_extracts_and_strips_content():
    data = {"choices": [{"message": {"content": "  Contact your clinic.  "}}]}
    assert parse_response(data) == "Contact your clinic."


def test_parse_response_rejects_malformed_payload():
    for bad in ({}, {"choices": []}, {"choices": [{"message": {"content": "  "}}]}):
        try:
            parse_response(bad)
        except ValueError:
            continue
        raise AssertionError("malformed payloads must be rejected: %r" % (bad,))


def test_ask_uses_injected_transport():
    seen = {}

    def transport(url, payload, timeout):
        seen["url"] = url
        seen["payload"] = payload
        seen["timeout"] = timeout
        return {"choices": [{"message": {"content": " Apply a mask and rest. "}}]}

    answer = ask("I have a cough", base_url="http://localhost:30000/v1/", transport=transport)

    assert answer == "Apply a mask and rest."
    assert seen["url"] == "http://localhost:30000/v1/chat/completions"
    assert seen["payload"]["model"] == MODEL_ID
    assert seen["payload"]["max_tokens"] == MAX_NEW_TOKENS
    assert seen["payload"]["messages"][-1]["content"] == "I have a cough"
    assert seen["payload"]["messages"][0]["content"] == health_chatbot.SYSTEM_PROMPT
