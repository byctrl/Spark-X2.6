"""Minimal health-information chatbot built on Spark-X2.6-4B.

The model is served behind an OpenAI-compatible endpoint (vLLM / SGLang /
llama.cpp server, see the README quickstart). This module is a small demo for
the Spark-X2.6 AI-for-Good challenge: it gives plain-language *education*
only, never a diagnosis or a treatment plan, and always points the user to a
qualified clinician.

Usage::

    python src/health_chatbot.py                  # built-in demo question
    python src/health_chatbot.py "my question"    # custom question

Environment variables::

    SPARK_BASE_URL   default http://127.0.0.1:30000/v1
    SPARK_API_KEY    default EMPTY
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 120

BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1").rstrip("/")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people with low health literacy.\n"
    "Rules:\n"
    "1. Explain in plain, simple language: short sentences, no medical jargon.\n"
    "2. Do NOT diagnose, do NOT prescribe, do NOT give medicine names or dosages.\n"
    "3. Never tell the user what disease they have.\n"
    "4. If something may need urgent care, say so clearly and tell the user to "
    "contact a qualified clinician or the local emergency number.\n"
    "5. Do not invent facts. If you are unsure, say that you are unsure.\n"
    "6. Answer in the same language as the user's question.\n"
    "7. Keep the answer under 250 words and end with one concrete next step."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and a fever in the evening. I do not have much money and I am not sure "
    "whether I should see a doctor. What should I know?"
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "Please consult a qualified clinician."
)


def build_messages(question: str) -> list[dict[str, str]]:
    """Build the bounded chat messages sent to Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_payload(
    question: str,
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """Build an OpenAI-compatible chat-completions request body."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return {
        "model": model,
        "messages": build_messages(question),
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def extract_answer(body: dict) -> str:
    """Pull the assistant text out of a chat-completions response."""
    choices = (body or {}).get("choices") or []
    if not choices:
        raise ValueError("model response contained no choices")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise ValueError("model response contained empty content")
    return content


def chat(
    question: str,
    base_url: str = BASE_URL,
    api_key: str = API_KEY,
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    """Send one question to the local Spark-X2.6-4B server and return the answer."""
    payload = build_payload(question, model=model, max_tokens=max_tokens)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return extract_answer(body)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    question = " ".join(argv).strip() or DEMO_QUESTION

    print(f"Model: {MODEL_ID}")
    print(f"Endpoint: {BASE_URL}")
    print(f"Question: {question}\n")

    try:
        answer = chat(question)
    except (urllib.error.URLError, OSError) as exc:
        print(f"Could not reach the Spark endpoint at {BASE_URL}: {exc}")
        print(
            "Start a local OpenAI-compatible Spark-X2.6-4B server "
            "(see the README quickstart) and try again."
        )
        return 1
    except ValueError as exc:
        print(f"Invalid request or response: {exc}")
        return 1

    print(answer)
    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
