"""Minimal health-information chatbot powered by Spark-X2.6-4B.

Spark-X2.6-4B is a text-only model: it does not accept images. This module
sends one user question to a locally hosted OpenAI-compatible Spark endpoint
(SGLang, vLLM, llama.cpp server, Spark-MLX-LLM, ...) and turns the reply into a
plain-language, education-only answer.

The model is not a clinician. The bot never diagnoses, never prescribes and
always points the reader to a qualified professional.
"""

from __future__ import annotations

import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
SPARK_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people with low health literacy. "
    "Answer in plain language, short sentences, and explain any unavoidable term. "
    "You do NOT diagnose, you do NOT prescribe, and you do NOT give dosages. "
    "Give general, widely accepted information only, list the warning signs that "
    "need urgent care, and always tell the reader to confirm with a qualified "
    "clinician or pharmacist. If the question needs a diagnosis or a treatment "
    "decision, say so and refer the reader onward. Keep the answer under 200 words."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak tiga hari lalu, dua hari setelah "
    "pulang dari perjalanan. Apa yang sebaiknya saya lakukan?"
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "For a diagnosis or treatment, contact a qualified clinician."
)

EMERGENCY_NOTICE = (
    "Some of what you describe can be an emergency. Contact local emergency "
    "services or go to the nearest clinic now."
)

RED_FLAGS = (
    "chest pain",
    "difficulty breathing",
    "cannot breathe",
    "shortness of breath",
    "severe bleeding",
    "unconscious",
    "seizure",
    "suicidal",
    "stroke",
    "sesak napas",
    "nyeri dada",
)


def build_messages(question: str) -> list:
    """Return the chat messages sent to Spark for one question."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def build_payload(messages: list, model: str = MODEL_ID, max_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Return an OpenAI-compatible chat-completions payload."""
    if not messages:
        raise ValueError("messages must not be empty")
    return {
        "model": model,
        "messages": list(messages),
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
        "stream": False,
    }


def extract_reply(response: dict) -> str:
    """Return the assistant text, or raise ValueError on a malformed response."""
    if not isinstance(response, dict):
        raise ValueError("response must be a JSON object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content") or ""
    if not content.strip():
        raise ValueError("response has an empty message content")
    return content.strip()


def has_red_flag(text: str) -> bool:
    """True when the text mentions an emergency warning sign."""
    lowered = (text or "").lower()
    return any(flag in lowered for flag in RED_FLAGS)


def call_model(
    question: str,
    base_url: str = SPARK_BASE_URL,
    model: str = MODEL_ID,
    timeout: float = 120.0,
) -> str:
    """Send one question to a local Spark OpenAI-compatible endpoint."""
    payload = build_payload(build_messages(question), model=model)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return extract_reply(json.loads(body))


def answer(question: str, generate=call_model) -> str:
    """Return the model answer plus the safety notices shown to the reader."""
    text = (generate(question) or "").strip()
    if not text:
        raise ValueError("model returned an empty answer")
    parts = [text]
    if has_red_flag(question):
        parts.append(EMERGENCY_NOTICE)
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def main(argv=None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    print(answer(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
