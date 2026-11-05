"""Minimal health-information chatbot demo powered by Spark-X2.6-4B.

Public-interest (health literacy) demo for HER Hack-Astron #7.
The script builds a bounded, safety-oriented prompt, sends it to a locally
hosted Spark-X2.6-4B OpenAI-compatible endpoint, and validates the reply.
It uses only the Python standard library.

Usage:
    python src/health_chatbot.py
    python src/health_chatbot.py "your question"

Environment:
    SPARK_BASE_URL  OpenAI-compatible base URL (default http://127.0.0.1:30000/v1)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
DEFAULT_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in plain, "
    "respectful language that a person without medical training can understand. "
    "You do NOT diagnose, prescribe, or recommend treatments, medicines or dosages. "
    "Always say that this is general education, not medical advice, and tell the user "
    "to consult a qualified clinician, or emergency services for urgent symptoms. "
    "If the question is unclear, ask one short clarifying question. "
    "Keep answers short: at most 6 sentences or 6 bullet points."
)

DISCLAIMER = (
    "This demo provides general health education only. It is not medical advice, "
    "diagnosis or treatment. Always consult a qualified clinician."
)

DEMO_QUESTION = (
    "I have had a dry cough and a mild fever for four days since returning from a "
    "trip. What should I know before I see a doctor?"
)


def build_messages(
    question: str, history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Build the chat messages for one bounded turn."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("history turns must be dicts")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError("history turns need role user/assistant and string content")
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(
    messages: List[Dict[str, str]],
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Build the OpenAI-compatible request body."""
    if not messages:
        raise ValueError("messages must not be empty")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }


def extract_answer(body: Dict[str, Any]) -> str:
    """Validate the endpoint response and return the assistant text."""
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response from Spark endpoint") from exc

    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def chat(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    base_url: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Ask Spark-X2.6-4B one question and return the validated answer."""
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    payload = build_payload(build_messages(question, history))
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"could not reach Spark endpoint at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Spark endpoint returned invalid JSON") from exc

    return extract_answer(body)


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION

    print(f"model: {MODEL_ID}")
    print(f"endpoint: {DEFAULT_BASE_URL}")
    print(f"question: {question}")
    print("-" * 60)

    try:
        answer = chat(question)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        print("start a local Spark-X2.6-4B OpenAI-compatible server and retry.")
        return 1

    print(answer)
    print("-" * 60)
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
