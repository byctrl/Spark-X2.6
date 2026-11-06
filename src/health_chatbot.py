"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

A safety-bounded patient-education assistant: it explains health topics in
plain language and never diagnoses, prescribes, or replaces a clinician.

The model is not bundled here. Serve Spark-X2.6-4B behind an
OpenAI-compatible endpoint, for example:

    python -m sglang.launch_server --model-path XHToken/Spark-X2.6-4B --port 30000

Then run:

    python src/health_chatbot.py "Saya batuk dan demam setelah pulang bepergian"

Set SPARK_BASE_URL to point at a different endpoint (default
http://127.0.0.1:30000/v1).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
TIMEOUT_SECONDS = 120
MAX_QUESTION_CHARS = 4000

SYSTEM_PROMPT = (
    "You are a patient-education assistant powered by Spark-X2.6-4B. "
    "Explain health topics in plain language for people with limited medical "
    "literacy, in the language the user writes in.\n"
    "Hard rules:\n"
    "1. Never diagnose, never prescribe, never give drug dosages, and never tell "
    "the user to stop a treatment a clinician prescribed.\n"
    "2. Do not claim to be a clinician and do not present your answer as medical "
    "advice.\n"
    "3. Separate what is generally known from what needs a professional check, "
    "and say clearly when information is uncertain.\n"
    "4. Always end with a short 'When to seek care' list of red-flag signs and "
    "advise contacting a qualified clinician or local emergency service.\n"
    "5. If the user describes an emergency, tell them to contact emergency "
    "services now instead of waiting for this chat.\n"
    "6. Never ask for or repeat personal identifiers such as full name, ID "
    "number, address, or phone number.\n"
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak tiga hari lalu, setelah pulang "
    "bepergian. Apa yang sebaiknya saya perhatikan, dan kapan harus ke dokter?"
)

DISCLAIMER = (
    "Education only - not a diagnosis or a prescription. "
    "Contact a qualified clinician or local emergency service for personal care."
)


class ChatbotError(RuntimeError):
    """Raised when the Spark endpoint is unreachable or returns unusable output."""


def build_messages(question: str) -> list[dict[str, str]]:
    """Build the chat messages for one bounded patient-education question."""
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    text = question.strip()
    if not text:
        raise ValueError("question must not be empty")
    if len(text) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"question must be at most {MAX_QUESTION_CHARS} characters"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def build_payload(
    question: str,
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """Build the OpenAI-compatible request body sent to Spark-X2.6-4B."""
    return {
        "model": model,
        "messages": build_messages(question),
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }


def extract_reply(payload: object) -> str:
    """Return the assistant text from a chat-completions payload."""
    if not isinstance(payload, dict):
        raise ChatbotError("endpoint returned a non-object payload")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ChatbotError("endpoint returned no choices")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content") or first.get("text") or ""
    if not isinstance(content, str) or not content.strip():
        raise ChatbotError("endpoint returned an empty answer")
    return content.strip()


def ask(
    question: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = TIMEOUT_SECONDS,
) -> str:
    """Send one question to the Spark-X2.6-4B endpoint and return its answer."""
    payload = build_payload(question)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as exc:
        raise ChatbotError(
            f"could not reach the Spark endpoint at {base_url}: {exc}"
        ) from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ChatbotError("endpoint returned invalid JSON") from exc
    return extract_reply(data)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    base_url = os.environ.get("SPARK_BASE_URL", DEFAULT_BASE_URL)
    try:
        answer = ask(question, base_url=base_url)
    except (ChatbotError, ValueError) as exc:
        print(f"[health_chatbot] {exc}", file=sys.stderr)
        print(
            "[health_chatbot] Serve XHToken/Spark-X2.6-4B behind an "
            f"OpenAI-compatible endpoint (default {DEFAULT_BASE_URL}) and "
            "optionally set SPARK_BASE_URL.",
            file=sys.stderr,
        )
        return 1
    print("Q:", question)
    print()
    print(answer)
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
