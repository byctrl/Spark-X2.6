"""Minimal Spark-X2.6-4B health-knowledge chatbot (HER Hack-Astron #7, AI for Good).

The application answers general health questions in plain language for people who
otherwise lack access to understandable information. It never diagnoses or
prescribes, and it always points the reader to a qualified clinician.

Layers:
  * pure helpers (build_messages / build_payload / parse_completion /
    ensure_disclaimer) - no model needed, unit tested in test_health_chatbot.py
  * generate() - calls a locally hosted, OpenAI-compatible Spark-X2.6-4B endpoint
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

DISCLAIMER = (
    "This is general health education, not a diagnosis or a prescription. "
    "Please consult a qualified clinician about your own situation."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant that helps people understand general "
    "health topics in simple, respectful language. Rules you must follow:\n"
    "1. Explain in plain words, short sentences, no jargon unless you define it.\n"
    "2. Never diagnose, never name a likely disease as the user's condition, and "
    "never prescribe medicines, dosages or treatments.\n"
    "3. Say clearly when something needs in-person care, and tell the reader to "
    "contact a qualified clinician or local emergency service.\n"
    "4. Do not ask for or repeat personal identifiers, and remind the reader not "
    "to share medical records in chat.\n"
    "5. If the question is outside general health education, say so and suggest a "
    "better source of help.\n"
    "6. Be inclusive: do not assume gender, age, ability, language or income."
)

DEMO_QUESTION = (
    "I came back from a trip three days ago and now I have a cough and a mild "
    "fever. What general things should I know, and when should I see a doctor?"
)


def build_messages(question: str, history=None):
    """Return the chat messages sent to Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            raise ValueError("history turns need role user/assistant and content")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_payload(messages, max_new_tokens=MAX_NEW_TOKENS):
    """Return the OpenAI-compatible request body for the local Spark server."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.3,
        "top_p": 0.9,
    }


def parse_completion(data):
    """Extract the assistant text from a chat-completion response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected completion payload: %r" % (data,)) from exc
    content = (content or "").strip()
    if not content:
        raise ValueError("model returned an empty answer")
    return content


def ensure_disclaimer(answer: str) -> str:
    """Append the safety disclaimer exactly once."""
    text = (answer or "").strip()
    if not text:
        raise ValueError("answer is empty")
    if DISCLAIMER in text:
        return text
    return text + "\n\n" + DISCLAIMER


def generate(question: str, history=None, base_url=None, max_new_tokens=MAX_NEW_TOKENS) -> str:
    """Ask the local Spark-X2.6-4B endpoint and return a safe, plain answer."""
    base = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    payload = build_payload(build_messages(question, history), max_new_tokens)
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("SPARK_API_KEY", "EMPTY"),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "cannot reach %s at %s (%s). Start a local OpenAI-compatible server, "
            "for example: python -m sglang.launch_server --model-path %s --port 30000"
            % (MODEL_ID, base, exc, MODEL_ID)
        ) from exc
    return ensure_disclaimer(parse_completion(data))


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    print("Q: " + question)
    print()
    print(generate(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
