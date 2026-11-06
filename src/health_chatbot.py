"""Minimal health-information chatbot built on Spark-X2.6-4B.

Public-interest demo for HER Hack-Astron #7 (AI for Good).

The model is served by a *local* OpenAI-compatible endpoint
(SGLang, vLLM, llama.cpp server, Ollama, ...). Nothing is sent to a cloud
API, so a community health worker can run the assistant offline.

What the model does here: it turns a plain-language question into a
plain-language patient-education answer. It is explicitly NOT allowed to
diagnose, prescribe or give doses; the application code adds an emergency
notice on top of the model output, and a human clinician remains the
authority.

Usage:
    python src/health_chatbot.py                 # run the demo question
    python src/health_chatbot.py -q "my question"

Environment:
    SPARK_BASE_URL  default http://127.0.0.1:30000/v1
    SPARK_API_KEY   default EMPTY (local servers usually ignore it)
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.2
REQUEST_TIMEOUT = 120

BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")

SYSTEM_PROMPT = (
    "You are a patient-education assistant that helps people understand "
    "general health information in plain language.\n\n"
    "Follow these rules:\n"
    "1. Use simple, everyday words and short sentences. Avoid medical jargon; "
    "if a term is needed, explain it in one short clause.\n"
    "2. Do NOT diagnose, do NOT name a disease as the cause, do NOT "
    "prescribe or recommend specific medicines, doses or treatments.\n"
    "3. Explain what is generally known, which signs are worth watching, and "
    "when a person should contact a clinician.\n"
    "4. If the question suggests an emergency (for example chest pain, "
    "trouble breathing, heavy bleeding, fainting, confusion, suicidal "
    "thoughts, or bleeding in pregnancy), tell the reader to seek emergency "
    "care right now, before anything else.\n"
    "5. State clearly that your answer is general information, not medical "
    "advice, and that only a qualified clinician can assess their situation.\n"
    "6. If important details are missing, ask one short clarifying question "
    "instead of guessing.\n"
    "7. Never ask for or repeat identifying details such as full name, "
    "address, ID number or insurance number."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago. For three days I have had a "
    "cough and a fever around 38.5 C. What can I do at home, and when "
    "should I see a doctor?"
)

EMERGENCY_KEYWORDS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "difficulty breathing",
    "shortness of breath",
    "heavy bleeding",
    "unconscious",
    "not breathing",
    "seizure",
    "stroke",
    "suicidal",
    "suicide",
    "bleeding in pregnancy",
)

EMERGENCY_NOTICE = (
    "URGENT: this may be an emergency. Please contact your local emergency "
    "number or go to the nearest emergency department now. The information "
    "below is general education only and cannot replace an examination."
)


def build_messages(question, history=None):
    """Build the OpenAI-style message list sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("history turns must be objects with role and content")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            raise ValueError("history turns need role user/assistant and non-empty content")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages):
    """Wrap messages in a bounded, non-streaming chat-completions payload."""
    if not messages:
        raise ValueError("messages must not be empty")
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text, or fail loudly on an unexpected shape."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected chat-completions response: %s" % exc) from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty message")
    return content.strip()


def needs_emergency_notice(question):
    """Application-side safety check, independent of the model output."""
    text = (question or "").lower()
    return any(keyword in text for keyword in EMERGENCY_KEYWORDS)


def chat(question, history=None, base_url=None, timeout=REQUEST_TIMEOUT):
    """Send one question to the local Spark-X2.6-4B endpoint and return text."""
    payload = build_payload(build_messages(question, history))
    url = (base_url or BASE_URL).rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "cannot reach the Spark endpoint at %s (%s). Start a local "
            "OpenAI-compatible server (see README) or set SPARK_BASE_URL." % (url, exc)
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("endpoint did not return JSON: %s" % exc) from exc

    answer = parse_response(data)
    if needs_emergency_notice(question):
        return EMERGENCY_NOTICE + "\n\n" + answer
    return answer


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a general health question (local endpoint)."
    )
    parser.add_argument("-q", "--question", default=DEMO_QUESTION)
    args = parser.parse_args(argv)

    print("model:", MODEL_ID)
    print("endpoint:", BASE_URL)
    print("question:", args.question)
    print("---")
    try:
        print(chat(args.question))
    except (RuntimeError, ValueError) as exc:
        print("error:", exc)
        print("This demo needs a local OpenAI-compatible server serving %s." % MODEL_ID)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
