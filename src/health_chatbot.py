"""Minimal patient-education chatbot demo for HER Hack-Astron #7.

Public-interest scenario: inclusive access to understandable health
information. The chatbot talks to Spark-X2.6-4B served by any
OpenAI-compatible local endpoint (vLLM, SGLang, llama.cpp server or
Spark-MLX-LLM) and explains health information in plain language.

The assistant is education-only: it never diagnoses, never prescribes and
always points people to a qualified clinician for personal decisions. A
human clinician remains the decision maker.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")
TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in "
    "simple, respectful language a person without medical training can "
    "understand. Do not diagnose. Do not prescribe, do not give dosages, and "
    "do not tell the user to start, stop or change any treatment. When a "
    "question needs a personal medical decision, say clearly that only a "
    "qualified clinician can make it, and point the user to a local clinic or "
    "emergency service. Mention warning signs that need urgent care when they "
    "are relevant. Say when you are unsure, because you can be wrong. Reply in "
    "the language of the user."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a cough and a mild "
    "fever in the evenings. What could it be, and when should I see someone?"
)


def build_messages(question: str, system_prompt: str = SYSTEM_PROMPT) -> list:
    """Build the chat messages sent to the model."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def build_payload(
    question: str,
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.3,
) -> dict:
    """Build the OpenAI-compatible chat-completions request body."""
    return {
        "model": model,
        "messages": build_messages(question),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def parse_reply(data: dict) -> str:
    """Extract the assistant text and reject empty or malformed answers."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected response shape from the model endpoint")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def ask(
    question: str,
    base_url: str = BASE_URL,
    model: str = MODEL_ID,
    timeout: float = TIMEOUT_SECONDS,
    opener=urllib.request.urlopen,
) -> str:
    """Send one question to Spark-X2.6-4B and return the plain-language answer."""
    payload = build_payload(question, model=model)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        },
        method="POST",
    )
    with opener(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return parse_reply(json.loads(raw))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B patient-education chatbot demo"
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args(argv)
    print(ask(args.question, base_url=args.base_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
