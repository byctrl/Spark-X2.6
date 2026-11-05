"""Minimal health-information chatbot built on Spark-X2.6-4B.

Public-interest demo (health + accessibility, HER Hack-Astron #7).
The assistant explains health topics in plain language, never diagnoses or
prescribes, and always routes the user to a qualified clinician.

It talks to a local OpenAI-compatible Spark-X2.6-4B endpoint (vLLM / SGLang /
Ollama):

    export SPARK_BASE_URL=http://127.0.0.1:30000/v1
    python src/health_chatbot.py "I have a cough and fever after travelling"

This is an educational demo, not medical advice.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

DISCLAIMER = (
    "This is general health education, not a diagnosis or treatment plan. "
    "Please consult a qualified clinician about your own situation."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people with limited access to "
    "health information. Explain in plain, respectful language at roughly a "
    "6th-grade reading level. Use short sentences and define medical terms.\n"
    "Rules:\n"
    "1. Never diagnose, never prescribe, never give drug dosages.\n"
    "2. Say clearly when something needs urgent professional or emergency care, "
    "and name the warning signs.\n"
    "3. If information is missing, say what a clinician would need to know.\n"
    "4. Do not ask for names, addresses, ID numbers, or other identifying data.\n"
    "5. End by recommending a qualified clinician for personal decisions."
)

DEMO_QUESTION = (
    "I have had a dry cough and mild fever for four days, and I returned from a "
    "trip abroad last week. What should I know before I see a doctor?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6 for one question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("history entries must be dicts with role and content")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant"):
            raise ValueError("history role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("history content must be a non-empty string")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages, model=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body."""
    if not messages:
        raise ValueError("messages must not be empty")
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.3,
        "top_p": 0.9,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text from a chat-completion response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected response shape from Spark-X2.6")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Spark-X2.6 returned an empty answer")
    return content.strip()


def with_disclaimer(text):
    """Append the safety footer once."""
    if DISCLAIMER in text:
        return text
    return text.rstrip() + "\n\n" + DISCLAIMER


def endpoint_url(base_url=None):
    """Resolve the chat-completions URL, honouring SPARK_BASE_URL."""
    base = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    return base + "/chat/completions"


def ask(question, history=None, base_url=None, model=MODEL_ID, timeout=REQUEST_TIMEOUT):
    """Send one question to a local Spark-X2.6-4B server and return the answer."""
    payload = build_payload(build_messages(question, history), model=model)
    request = urllib.request.Request(
        endpoint_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return with_disclaimer(parse_response(data))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    try:
        answer = ask(question)
    except (ValueError, OSError) as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        print("hint: serve {0} and set SPARK_BASE_URL".format(MODEL_ID), file=sys.stderr)
        return 1
    print("Q: " + question)
    print()
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
