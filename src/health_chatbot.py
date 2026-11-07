"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

A small, dependency-free demo that sends a plain-language health question to a
local Spark-X2.6-4B OpenAI-compatible endpoint (SGLang / vLLM / llama.cpp server)
and prints the answer.

The assistant is scoped to patient education: it explains terms in simple
language and suggests questions for a clinician. It never diagnoses, prescribes
or claims to replace in-person care.

Usage:
    python src/health_chatbot.py --demo
    python src/health_chatbot.py "What does high blood pressure mean?"
    python src/health_chatbot.py --base-url http://127.0.0.1:30000/v1 --demo
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Mapping, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people with limited access to "
    "health information. Answer in the language the user writes in, and explain "
    "medical terms in simple, everyday words.\n"
    "Rules:\n"
    "1. Do not diagnose, do not tell the user which disease they have, and do "
    "not recommend prescription drugs, dosages or treatments.\n"
    "2. Give general, widely accepted information and list the questions the "
    "user should ask a qualified clinician.\n"
    "3. Say clearly when a symptom needs urgent in-person care (for example "
    "chest pain, difficulty breathing, heavy bleeding, fainting, confusion).\n"
    "4. If important facts are missing (age, how long, how severe), ask for "
    "them instead of guessing.\n"
    "5. Never claim to replace a doctor. Keep the answer short and structured."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and a fever that comes back every evening. There is no doctor near my "
    "town. What could this be and what should I do?"
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "Please see a qualified clinician for your own situation."
)


def build_messages(
    question: str,
    history: Sequence[Mapping[str, str]] = (),
) -> list[dict[str, str]]:
    """Build the chat messages for one bounded education turn."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(
    messages: Sequence[Mapping[str, str]],
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> dict:
    """Wrap messages in an OpenAI-compatible chat-completions payload."""
    return {
        "model": model,
        "messages": [dict(message) for message in messages],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }


def parse_response(data: Mapping) -> str:
    """Extract the answer text, rejecting empty or malformed responses."""
    choices = data.get("choices") if isinstance(data, Mapping) else None
    if not choices:
        raise ValueError("response contains no choices")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response contains empty content")
    return content.strip()


def ask(
    question: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = MODEL_ID,
    history: Sequence[Mapping[str, str]] = (),
    timeout: int = REQUEST_TIMEOUT,
    opener=urllib.request.urlopen,
) -> str:
    """Send one question to the local Spark-X2.6-4B server and return the text."""
    payload = build_payload(build_messages(question, history), model=model)
    request = urllib.request.Request(
        url=base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return parse_response(json.loads(body))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a plain-language health question."
    )
    parser.add_argument("question", nargs="*", help="your health question")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the built-in demo scenario instead of your own question",
    )
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip()
    if args.demo or not question:
        question = DEMO_QUESTION

    print(f"model:    {args.model}")
    print(f"endpoint: {args.base_url}")
    print(f"question: {question}\n")

    try:
        answer = ask(question, base_url=args.base_url, model=args.model)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"Could not get an answer: {exc}")
        print("Start a Spark-X2.6-4B OpenAI-compatible server or pass --base-url.")
        return 1

    print(answer)
    print(f"\n---\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
