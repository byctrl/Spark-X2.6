"""Minimal health-information chatbot powered by Spark-X2.6-4B.

The application keeps the model behind a small, dependency-free HTTP client so
the demo can run against a local OpenAI-compatible Spark-X2.6 endpoint.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from urllib.request import Request, urlopen

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1").rstrip("/")
REQUEST_TIMEOUT = 60

DISCLAIMER = (
    "This information is for general education only and is not medical advice. "
    "It cannot diagnose or treat a condition. Please consult a qualified health "
    "professional for personal medical concerns."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in plain, "
    "inclusive language that is easy to understand. Do not diagnose, prescribe, "
    "or recommend treatment. Encourage the user to contact a qualified clinician. "
    "If symptoms could be urgent, tell the user to seek emergency care immediately."
)

DEMO_QUESTION = (
    "I have had a cough and fever since I returned from a trip three days ago. "
    "What questions should I ask a doctor, and when should I seek urgent care?"
)


class ChatbotError(RuntimeError):
    """Raised when the model endpoint cannot produce a usable answer."""


def build_messages(question: str) -> List[Dict[str, str]]:
    """Build the bounded chat messages sent to Spark-X2.6."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_payload(question_or_messages: Any) -> Dict[str, Any]:
    """Build an OpenAI-compatible chat-completions payload."""
    if isinstance(question_or_messages, str):
        messages = build_messages(question_or_messages)
    else:
        messages = list(question_or_messages)
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
    }


def extract_text(response: Any) -> str:
    """Extract text from either chat-completion or text-completion shapes."""
    if isinstance(response, (bytes, bytearray)):
        response = response.decode("utf-8")
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except (TypeError, ValueError) as exc:
            raise ChatbotError("model endpoint returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise ChatbotError("model endpoint returned an invalid response")

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ChatbotError("model response did not contain choices")

    first = choices[0]
    if not isinstance(first, dict):
        raise ChatbotError("model response contained an invalid choice")

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            text = "".join(parts).strip()
            if text:
                return text
        raise ChatbotError("model response contained empty content")

    text = first.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    raise ChatbotError("model response contained no usable text")


def _read_response(response: Any) -> str:
    """Read a response object, including simple test doubles."""
    raw = response.read()
    if not isinstance(raw, (bytes, bytearray, str)):
        enter = getattr(response, "__enter__", None)
        if enter is None:
            raise ChatbotError("model endpoint returned an unreadable response")
        with response as entered:
            raw = entered.read()
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8")
    if isinstance(raw, str):
        return raw
    raise ChatbotError("model endpoint returned an unreadable response")


def ask(question: str, base_url: str | None = None, timeout: int | None = None) -> str:
    """Send a question to the local Spark-X2.6 chat endpoint and return its answer."""
    endpoint = (base_url or BASE_URL).rstrip("/") + "/chat/completions"
    payload = build_payload(question)
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        try:
            response = urlopen(request, timeout=REQUEST_TIMEOUT if timeout is None else timeout)
        except TypeError:
            response = urlopen(request)
        raw = _read_response(response)
    except ChatbotError:
        raise
    except Exception as exc:
        raise ChatbotError(f"could not reach model endpoint: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ChatbotError("model endpoint returned invalid JSON") from exc

    return extract_text(parsed)


def format_answer(answer: str) -> str:
    """Attach the safety disclaimer to a model answer."""
    return f"{answer}\n\n{DISCLAIMER}"


def main() -> None:
    question = input("Question: ").strip() or DEMO_QUESTION
    try:
        print(format_answer(ask(question)))
    except ChatbotError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
