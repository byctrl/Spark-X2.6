"""Minimal Spark-X2.6-4B chatbot for inclusive health information.

The application is intentionally small: it turns a user question into a bounded
chat-completion request for a locally served Spark-X2.6-4B model, and prints a
plain-language patient-education answer. It never diagnoses, prescribes or
replaces a clinician, and it needs no third-party package.

Run (model served behind an OpenAI-compatible endpoint, e.g. SGLang/vLLM):

    python src/health_chatbot.py
    python src/health_chatbot.py 'Saya demam tiga hari, apa yang harus saya lakukan?'

Configure the endpoint with SPARK_BASE_URL (default http://127.0.0.1:30000/v1).
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who have limited access "
    "to health information. Explain in plain language and short sentences, and "
    "answer in the language the user writes in. "
    "You do NOT diagnose, prescribe, recommend dosages, or interpret test results. "
    "State clearly which warning signs mean the user should seek urgent care, and "
    "always remind the user to confirm with a qualified clinician. "
    "If the question is unclear, ask one short clarifying question first. "
    "Never claim to replace a doctor or an emergency service."
)

DEMO_QUESTION = (
    "I have had a dry cough and a fever of 38 C for three days, and I returned "
    "from a long trip last week. What should I do?"
)


def build_messages(question: str, history=None):
    """Build the chat messages for one question, keeping the system prompt first."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        if role not in ("user", "assistant"):
            raise ValueError("history turns must be 'user' or 'assistant'")
        messages.append({"role": role, "content": str(turn.get("content", ""))})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages, max_tokens: int = MAX_NEW_TOKENS):
    """Bounded OpenAI-compatible request body."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def extract_text(response):
    """Return the assistant text, or raise if the completion is unusable."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("malformed completion response") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty completion content")
    return content.strip()


def complete(messages, base_url: str = DEFAULT_BASE_URL, timeout: int = 120):
    """Send the messages to the local Spark-X2.6-4B endpoint and return the text."""
    payload = json.dumps(build_payload(messages)).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return extract_text(body)


def ask(question: str, history=None, base_url: str = DEFAULT_BASE_URL):
    """Convenience helper used by the CLI."""
    return complete(build_messages(question, history), base_url=base_url)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ask Spark-X2.6-4B a health question.")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args(argv)
    print(ask(args.question, base_url=args.base_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
