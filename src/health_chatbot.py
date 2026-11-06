"""Minimal Spark-X2.6-4B health-information chatbot.

Public-interest demo for the HER Hack-Astron #7 challenge: help people read
general health information in plain language. The assistant never diagnoses and
never prescribes; every answer points the person to a qualified clinician.

The model is called through an OpenAI-compatible endpoint (SGLang, vLLM,
llama.cpp or Ollama all expose one), so this module stays dependency-free and
runs fully locally.

Usage:
    python src/health_chatbot.py "I have had a dry cough for three days"

Environment:
    SPARK_BASE_URL  base URL of the local OpenAI-compatible server
                    (default: http://127.0.0.1:30000/v1)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Callable, Mapping, Optional, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
REQUEST_TIMEOUT = 120.0

SYSTEM_PROMPT = (
    "You are a public-interest health education assistant for people who may "
    "have limited access to medical information.\n"
    "Rules:\n"
    "1. Explain general health topics in plain, everyday language. Keep the "
    "answer short and clearly structured.\n"
    "2. Never diagnose a condition, never prescribe or dose medication, and "
    "never claim to replace a clinician.\n"
    "3. If the question suggests an emergency (chest pain, difficulty "
    "breathing, heavy bleeding, fainting, thoughts of self-harm), tell the "
    "person to contact local emergency services right away.\n"
    "4. State clearly when information is uncertain or when local guidance may "
    "differ, and finish by recommending that the person speak with a "
    "qualified clinician.\n"
    "5. Do not ask for or repeat names, addresses, or other identifying "
    "details."
)

DEMO_QUESTION = (
    "I have had a dry cough and a mild fever for three days and just came back "
    "from a trip abroad. What should I do?"
)

# A client takes (messages, max_new_tokens) and returns the model's raw text.
Client = Callable[[Sequence[Mapping[str, str]], int], str]


def build_messages(
    question: str,
    history: Optional[Sequence[Mapping[str, str]]] = None,
) -> list:
    """Build the chat payload: system rules, optional history, then the question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            raise ValueError(
                "history turns need role in {'user', 'assistant'} "
                "and non-empty content"
            )
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})
    return messages


def complete(
    messages: Sequence[Mapping[str, str]],
    max_new_tokens: int = MAX_NEW_TOKENS,
    base_url: Optional[str] = None,
    model: str = MODEL_ID,
    timeout: float = REQUEST_TIMEOUT,
) -> str:
    """Call the local Spark-X2.6 OpenAI-compatible /chat/completions endpoint."""
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "cannot reach the Spark-X2.6 endpoint at {}: {}".format(url, exc)
        ) from exc
    except ValueError as exc:
        raise RuntimeError("endpoint returned invalid JSON") from exc

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("unexpected response shape from endpoint") from exc


def chat(
    question: str,
    history: Optional[Sequence[Mapping[str, str]]] = None,
    client: Optional[Client] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Answer one question; ``client`` can be injected for tests or other runtimes."""
    messages = build_messages(question, history)
    generate = client or complete
    answer = (generate(messages, max_new_tokens) or "").strip()
    if not answer:
        raise RuntimeError("model returned an empty answer")
    return answer


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B plain-language health information chatbot"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEMO_QUESTION,
        help="question to ask (defaults to the demo question)",
    )
    args = parser.parse_args(argv)

    try:
        print(chat(args.question))
    except (RuntimeError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
