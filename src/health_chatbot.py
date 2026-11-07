"""Minimal health-information chatbot demo for the Spark-X2.6 "AI for Good" challenge.

Talks to a locally served Spark-X2.6-4B OpenAI-compatible endpoint (see the
repository README / FAQ for runtime instructions). Standard library only: no
third-party dependencies are required for the demo to run.

The bot is deliberately bounded: it explains health topics in plain language,
but never diagnoses, prescribes or replaces a clinician, and every answer ends
with a disclaimer pointing the user to a qualified professional.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health topics in plain "
    "language for a general audience. You do NOT provide diagnosis, "
    "prescriptions, dosages or treatment plans, and you never claim to replace "
    "a clinician. Always remind the user to consult a qualified health "
    "professional, and point to emergency services when the description sounds "
    "urgent. Keep answers short and say so when you are not confident."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and fever in the evening. What could it be and what should I do?"
)

DISCLAIMER = (
    "This is general health information, not medical advice. "
    "Please consult a qualified clinician."
)


def build_messages(question, history=None):
    """Build the chat message list sent to the model."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    text = (question or "").strip()
    if not text:
        raise ValueError("question must not be empty")
    messages.append({"role": "user", "content": text})
    return messages


def build_payload(question, history=None, model=MODEL_ID):
    """Build the OpenAI-compatible request body."""
    return {
        "model": model,
        "messages": build_messages(question, history),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text from an OpenAI-compatible response body."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response shape: %r" % (data,)) from exc
    text = (content or "").strip()
    if not text:
        raise ValueError("model returned an empty answer")
    return text


def format_reply(answer):
    """Append the safety disclaimer to a model answer."""
    return answer.rstrip() + "\n\n" + DISCLAIMER


def ask(question, history=None, base_url=None, model=MODEL_ID, timeout=TIMEOUT_SECONDS):
    """Send one question to the model and return the disclaimed answer."""
    url = (base_url or BASE_URL).rstrip("/") + "/chat/completions"
    body = json.dumps(build_payload(question, history, model)).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return format_reply(parse_response(data))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    try:
        print(ask(question))
    except (urllib.error.URLError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        print(
            "hint: serve %s locally and set SPARK_BASE_URL if the port differs" % MODEL_ID,
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
