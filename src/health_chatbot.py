"""Minimal health-information chatbot powered by Spark-X2.6-4B.

Patient-education assistant with a safety boundary:

* explains health information in plain language
* never diagnoses, prescribes or replaces a clinician
* always points the user to qualified professional care

The model is served through an OpenAI-compatible endpoint (SGLang, vLLM,
Ollama or Spark-MLX-LLM). Point the client at it with ``SPARK_BASE_URL`` or
``--base-url``.

Usage::

    python src/health_chatbot.py
    python src/health_chatbot.py "Saya batuk dan demam setelah bepergian"
    python src/health_chatbot.py --base-url http://127.0.0.1:30000/v1 "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in "
    "simple, respectful language that a person without medical training can "
    "understand.\n"
    "Rules:\n"
    "1. Never give a diagnosis and never prescribe or adjust medication.\n"
    "2. Do not claim to replace a doctor, nurse or pharmacist.\n"
    "3. Describe warning signs that mean the person should seek urgent care.\n"
    "4. If the question needs a clinical decision, say so and recommend a "
    "qualified clinician.\n"
    "5. Be concise, use short sentences, and avoid unexplained jargon.\n"
    "6. If information is missing, ask one clarifying question instead of "
    "guessing."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak tiga hari lalu, setelah "
    "bepergian. Apa yang sebaiknya saya perhatikan?"
)

DISCLAIMER = (
    "This is general health information for education only, not medical "
    "advice. Please consult a qualified clinician about your own situation."
)

DISCLAIMER_SENTENCE = "not medical advice"


class ChatbotError(RuntimeError):
    """Raised when the Spark endpoint cannot be used or returns bad output."""


def build_messages(question, history=None):
    """Return the chat messages sent to Spark for ``question``."""
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


def build_payload(question, history=None, max_new_tokens=MAX_NEW_TOKENS):
    """Return the OpenAI-compatible request body for the Spark endpoint."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": int(max_new_tokens),
        "temperature": TEMPERATURE,
        "stream": False,
    }


def with_disclaimer(text):
    """Ensure the answer ends with the education-only disclaimer."""
    text = (text or "").strip()
    if DISCLAIMER_SENTENCE in text:
        return text
    if not text:
        return DISCLAIMER
    return "%s\n\n%s" % (text, DISCLAIMER)


def parse_response(data):
    """Extract the assistant answer from an OpenAI-compatible response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatbotError("unexpected response from the Spark endpoint") from exc

    content = (content or "").strip()
    if not content:
        raise ChatbotError("Spark returned an empty answer")
    return with_disclaimer(content)


def ask(question, base_url=None, timeout=120.0, history=None):
    """Send ``question`` to Spark-X2.6-4B and return the plain-language answer."""
    base_url = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    payload = build_payload(question, history)
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except OSError as exc:
        raise ChatbotError(
            "cannot reach Spark at %s (%s); start a local OpenAI-compatible "
            "server or pass --base-url" % (base_url, exc)
        ) from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ChatbotError("Spark endpoint did not return JSON") from exc

    return parse_response(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health information chatbot")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    print("model:", MODEL_ID)
    print("question:", args.question)
    try:
        answer = ask(args.question, base_url=args.base_url, timeout=args.timeout)
    except (ChatbotError, ValueError) as exc:
        print("error:", exc, file=sys.stderr)
        return 1

    print("\nanswer:\n" + answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
