"""Minimal Spark-X2.6-4B chatbot for inclusive health information (AI for Good demo).

Answers general health-education questions in plain language. It does not
diagnose and does not prescribe: every reply reminds the reader to consult a
qualified clinician.

The model is served by a local OpenAI-compatible endpoint (see the repository
Quickstart and FAQ). Configuration comes from the environment:

    SPARK_BASE_URL   default http://127.0.0.1:30000/v1
    SPARK_API_KEY    default "EMPTY"
    SPARK_MODEL      default MODEL_ID

Run the built-in demo question:

    python src/health_chatbot.py

Ask your own question:

    python src/health_chatbot.py "How can I avoid dehydration?"
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
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health topics in plain "
    "language at roughly a 6th-grade reading level, in the language the user "
    "writes in. You do NOT diagnose, you do NOT prescribe and you never give "
    "dosages. You never claim to replace a clinician. Structure every answer "
    "as: (1) a short summary, (2) general background, (3) practical self-care "
    "and prevention steps, (4) clear red-flag signs that need urgent care. "
    "If the question suggests an emergency, say so first and tell the reader "
    "to contact local emergency services. End with a reminder to consult a "
    "qualified health professional who knows the person's situation."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago and since yesterday I "
    "have a dry cough, a mild fever and I feel tired. What should I know?"
)


def build_messages(question, history=None):
    """Build the chat messages for one turn, with an optional prior history."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        role = turn.get("role")
        if role not in ("user", "assistant"):
            raise ValueError("history roles must be 'user' or 'assistant'")
        messages.append({"role": role, "content": str(turn.get("content", ""))})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS, temperature=0.3):
    """Build the OpenAI-compatible request body."""
    if not messages:
        raise ValueError("messages must not be empty")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text from a chat-completions response."""
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("response has no choices") from exc

    message = choice.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response has no assistant content")
    return content.strip()


def ask(question, history=None, base_url=None, api_key=None, model=None,
        timeout=REQUEST_TIMEOUT, opener=None):
    """Send one question to the local Spark-X2.6 endpoint and return the reply."""
    base_url = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    api_key = api_key or os.environ.get("SPARK_API_KEY") or "EMPTY"
    model = model or os.environ.get("SPARK_MODEL") or MODEL_ID

    payload = build_payload(build_messages(question, history), model=model)
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )

    open_fn = opener or urllib.request.urlopen
    with open_fn(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return parse_response(data)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION

    print("model:", os.environ.get("SPARK_MODEL") or MODEL_ID)
    print("endpoint:", os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL)
    print("question:", question)
    print("-" * 60)

    try:
        print(ask(question))
    except (urllib.error.URLError, ValueError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        print(
            "hint: start a local OpenAI-compatible Spark-X2.6-4B server and "
            "point SPARK_BASE_URL at it",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
