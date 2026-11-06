"""Minimal Spark-X2.6-4B health-information chatbot.

Public-interest demo for the HER Hack-Astron #7 "AI for Good" challenge.
The assistant gives general patient education in plain language and never
diagnoses, prescribes, or suggests dosages. It talks to a locally hosted
Spark-X2.6-4B OpenAI-compatible endpoint (for example SGLang or vLLM on
http://127.0.0.1:30000/v1).

Nothing here is medical advice. A person with an urgent problem should
contact local emergency services or a clinician.
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
    "You are a patient-education assistant. Explain health topics in simple, "
    "respectful language for a general reader. You do not diagnose, you do "
    "not prescribe, and you do not give dosages; you never tell someone to "
    "start, stop, or change a treatment a clinician gave them. If the "
    "question suggests an emergency (chest pain, difficulty breathing, "
    "fainting, heavy bleeding, or thoughts of self-harm), tell the person to "
    "contact local emergency services or a crisis line right away. Otherwise, "
    "give general background, list questions the person can ask a qualified "
    "clinician, and remind them that this is education, not medical advice."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago and now I have a dry "
    "cough and a mild fever. What should I know before I see a doctor?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    ``history`` is an optional list of earlier {role, content} turns with
    role ``user`` or ``assistant``. The system prompt is always first and the
    new user question is always last.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("history turns must be objects")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant"):
            raise ValueError("history turns need role 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("history turns need non-empty content")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages, max_new_tokens=MAX_NEW_TOKENS, temperature=0.3):
    """Wrap messages in an OpenAI-compatible completion request body."""
    if not messages:
        raise ValueError("messages must not be empty")
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": False,
    }


def parse_response(payload):
    """Extract the assistant text from an OpenAI-compatible response."""
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response shape from Spark-X2.6-4B") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def complete(base_url, payload, timeout=REQUEST_TIMEOUT):
    """POST the payload to the local endpoint and return the answer text."""
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("endpoint did not return JSON") from exc
    return parse_response(data)


def ask(question, base_url=None, history=None):
    """Convenience helper: question in, plain-language answer out."""
    base_url = base_url or os.environ.get("SPARK_BASE_URL", DEFAULT_BASE_URL)
    messages = build_messages(question, history=history)
    return complete(base_url, build_payload(messages))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    base_url = os.environ.get("SPARK_BASE_URL", DEFAULT_BASE_URL)
    print("Model: " + MODEL_ID)
    print("Endpoint: " + base_url)
    print("Question: " + question)
    print("-" * 60)
    try:
        print(ask(question, base_url=base_url))
    except urllib.error.URLError as exc:
        print("Could not reach the Spark-X2.6-4B endpoint: %s" % exc, file=sys.stderr)
        print("Start a local OpenAI-compatible server and set SPARK_BASE_URL.", file=sys.stderr)
        return 2
    except ValueError as exc:
        print("Request failed: %s" % exc, file=sys.stderr)
        return 1
    print("-" * 60)
    print("Education only, not medical advice. Please consult a qualified clinician.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
