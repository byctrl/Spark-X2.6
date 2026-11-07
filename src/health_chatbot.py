"""Minimal Spark-X2.6-4B chatbot for inclusive health information.

Public-interest demo: answers general health questions in plain language and
always redirects the user to a clinician or a local emergency service. It
never diagnoses, never prescribes, and never asks for personal data.

Run it against a local OpenAI-compatible Spark-X2.6-4B endpoint::

    export SPARK_BASE_URL=http://127.0.0.1:30000/v1
    python src/health_chatbot.py
    python src/health_chatbot.py "my own question"

Only the Python standard library is required.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
MAX_QUESTION_CHARS = 4000
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120.0

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who need clear, "
    "inclusive health information. Answer in the language the user wrote in, "
    "with plain words and short sentences. You do not diagnose, you do not "
    "prescribe, and you do not replace a clinician: never name a diagnosis, "
    "never give dosages, and never propose a treatment plan. Instead, explain "
    "in general terms what the described situation may involve, list warning "
    "signs that need urgent care, and recommend contacting a qualified "
    "clinician or the local emergency service. If the question is not about "
    "health information, say so briefly. Do not ask for names, addresses, ID "
    "numbers, or any other identifying data, and do not repeat such data if "
    "the user volunteers it."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days since I came back from "
    "a trip. There is no doctor nearby. What should I watch for, and when "
    "should I seek urgent care?"
)


def build_messages(question, history=None):
    """Build the chat messages for one health question."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must not be empty")
    if len(text) > MAX_QUESTION_CHARS:
        raise ValueError(
            "question must be at most %d characters" % MAX_QUESTION_CHARS
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history or ():
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            raise ValueError(
                "history entries must be user or assistant turns with content"
            )
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": text})
    return messages


def build_payload(messages):
    """Build the OpenAI-compatible request body for Spark-X2.6-4B."""
    if not messages:
        raise ValueError("messages must not be empty")
    return {
        "model": MODEL_ID,
        "messages": list(messages),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text from a chat completion payload."""
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response contains no choices")

    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise ValueError("choice is missing a message")

    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")

    return content.strip()


def _post(base_url, payload, api_key, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(
            "Spark server returned HTTP %s: %s" % (exc.code, detail)
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "cannot reach Spark server at %s: %s" % (url, exc.reason)
        ) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Spark server returned invalid JSON") from exc


def ask(question, base_url=None, api_key=None, timeout=REQUEST_TIMEOUT, history=None):
    """Send one question to Spark-X2.6-4B and return the answer text."""
    messages = build_messages(question, history)
    payload = build_payload(messages)
    base_url = base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL
    if api_key is None:
        api_key = os.environ.get("SPARK_API_KEY")
    return parse_response(_post(base_url, payload, api_key, timeout))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION

    print("model: %s" % MODEL_ID)
    print("question: %s" % question)
    print("")

    try:
        answer = ask(question)
    except (RuntimeError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1

    print(answer)
    print("")
    print(
        "Reminder: this is general health information, not medical advice. "
        "Contact a qualified clinician or the local emergency service for care."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
