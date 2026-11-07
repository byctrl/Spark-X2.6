"""Minimal health-information chatbot demo for Spark-X2.6-4B.

Public-interest (health literacy) demo: answers general health questions in
plain language, in the user's own language, and always points people to a
real clinician. It never diagnoses and never prescribes.

The model runs behind an OpenAI-compatible Spark-X2.6 endpoint; nothing about
this file is model-specific beyond MODEL_ID and the chat-completions shape.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
DEFAULT_TIMEOUT = float(os.environ.get("SPARK_TIMEOUT", "120"))

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public health information line. "
    "Explain general health topics in simple, everyday language and in the same "
    "language the user writes in.\n"
    "Hard rules:\n"
    "1. You must never diagnose a condition; do not say what disease someone has.\n"
    "2. You must never recommend a medicine, dose, or treatment plan; do not "
    "tell anyone to start or stop a medication.\n"
    "3. If the question suggests an emergency (chest pain, trouble breathing, "
    "heavy bleeding, fainting, thoughts of self-harm), tell the person to contact "
    "local emergency services right away.\n"
    "4. Always end by encouraging the person to speak with a qualified clinician "
    "or a local health worker for their own situation.\n"
    "5. If you are unsure, say so plainly instead of guessing.\n"
    "You give general information only; the clinician decides care."
)

DISCLAIMER = (
    "This demo gives general health information only. It is not medical advice, "
    "diagnosis, or treatment. For anything urgent, contact local emergency "
    "services; otherwise speak with a qualified clinician."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak tiga hari lalu, dua hari setelah "
    "pulang dari perjalanan. Apa yang sebaiknya saya perhatikan sebelum ke klinik?"
)


def build_messages(question, history=None):
    """Build the chat-completions message list for one turn.

    ``history`` is an optional list of earlier ``{"role", "content"}`` turns;
    only user/assistant turns with non-empty content are kept.
    """

    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})
    return messages


def build_payload(messages, max_new_tokens=MAX_NEW_TOKENS):
    """Wrap messages in an OpenAI-compatible request body for Spark-X2.6-4B."""

    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.3,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text from a chat-completions response."""

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response payload from Spark endpoint") from exc

    text = (content or "").strip()
    if not text:
        raise ValueError("model returned an empty answer")
    return text


def ask(question, history=None, base_url=DEFAULT_BASE_URL, timeout=DEFAULT_TIMEOUT):
    """Send one question to a local Spark-X2.6 endpoint and return the reply."""

    payload = build_payload(build_messages(question, history))
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("SPARK_API_KEY", "EMPTY"),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "could not reach the Spark-X2.6 endpoint at "
            f"{base_url} ({exc}); start your local OpenAI-compatible server "
            "or set SPARK_BASE_URL"
        ) from exc

    return parse_response(data)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(arg for arg in argv if arg.strip()) or DEMO_QUESTION

    print(f"model: {MODEL_ID}")
    print(f"endpoint: {DEFAULT_BASE_URL}")
    print(f"question: {question}\n")
    print(ask(question))
    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
