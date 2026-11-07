"""Minimal health-knowledge chatbot built on Spark-X2.6-4B.

What the model does here: it answers a plain-language health-education question.
Spark-X2.6-4B is a text-only model (no native image input) served behind an
OpenAI-compatible endpoint (SGLang / vLLM / llama.cpp / Spark-MLX-LLM).

Safety boundary: the bot explains, it does not diagnose or prescribe. It tells
the user when to seek a clinician, and this module never treats the model output
as a medical decision.

Run the demo:

    python src/health_chatbot.py --demo

Ask your own question:

    python src/health_chatbot.py --question "How can I lower my risk of heat stroke?"

Point it at a different local server:

    SPARK_BASE_URL=http://127.0.0.1:8000/v1 python src/health_chatbot.py --demo
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
MAX_INPUT_CHARS = 4000
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120.0

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "It cannot diagnose a condition or prescribe treatment."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "knowledge chatbot. Explain health topics in short, plain sentences that "
    "someone without medical training can follow.\n"
    "Rules you must follow:\n"
    "1. Never diagnose a condition and never recommend a prescription drug, "
    "a dosage, or a treatment plan.\n"
    "2. Explain what is commonly known, what to watch for, and when to seek "
    "care.\n"
    "3. When the situation could be urgent, say clearly that the person should "
    "contact a qualified clinician or a local emergency service now.\n"
    "4. Do not ask for or repeat identifying personal details.\n"
    "5. If the question is outside general health education, say so briefly.\n"
    "6. End every answer with this exact sentence: " + DISCLAIMER
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and now I have a cough and a mild "
    "fever. What should I pay attention to?"
)


def build_messages(question: str) -> list[dict[str, str]]:
    """Build the system + user messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    cleaned = " ".join(question.split())
    if len(cleaned) > MAX_INPUT_CHARS:
        cleaned = cleaned[:MAX_INPUT_CHARS]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": cleaned},
    ]


def build_payload(question: str, model: str = MODEL_ID) -> dict:
    """Build an OpenAI-compatible chat-completions payload."""
    return {
        "model": model,
        "messages": build_messages(question),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
    }


def extract_answer(response: dict) -> str:
    """Pull the assistant text out of an OpenAI-compatible response."""
    choices = response.get("choices") if isinstance(response, dict) else None
    if not choices:
        raise ValueError("response contains no choices")
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    content = content.strip()
    if not content:
        raise ValueError("response contains no assistant content")
    return content


def chat(
    question: str,
    base_url: str | None = None,
    timeout: float | None = None,
) -> str:
    """Send one question to the local Spark-X2.6-4B endpoint and return the answer."""
    payload = build_payload(question)
    endpoint = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = endpoint + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout or REQUEST_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach Spark endpoint at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Spark endpoint at {url} returned invalid JSON") from exc
    return extract_answer(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a plain-language health-education question."
    )
    parser.add_argument("--question", help="question to ask")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo question")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL of the local Spark server",
    )
    args = parser.parse_args(argv)

    if args.question:
        question = args.question
    elif args.demo:
        question = DEMO_QUESTION
    else:
        parser.error("pass --question TEXT or --demo")
        return 2

    print(f"Q: {question}\n")
    try:
        print(chat(question, base_url=args.base_url))
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
