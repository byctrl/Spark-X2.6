"""Minimal Spark-X2.6-4B health-information chatbot demo.

Submission aid for HER Hack-Astron #7 (Spark-X2.6 Open Source AI for Good).
The model is used only to explain general health information in plain
language. It never diagnoses or prescribes; every answer ends with a note
pointing the reader to a qualified clinician.

Usage:
    export SPARK_BASE_URL=http://127.0.0.1:30000/v1
    python src/health_chatbot.py "my question"

SPARK_BASE_URL must point at an OpenAI-compatible server hosting
XHToken/Spark-X2.6-4B (see README quickstart).
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

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain general health "
    "information in simple, everyday language for someone with no medical "
    "training. Keep answers short and concrete. You must not diagnose a "
    "condition, name a likely disease, recommend or adjust any medicine, or "
    "give a treatment plan. State clearly which warning signs mean the person "
    "should seek urgent care, and always tell the reader to consult a "
    "qualified clinician about their own situation. If the question falls "
    "outside general health education, say so and suggest an appropriate "
    "professional or service instead."
)

SAFETY_NOTE = (
    "This is general education only, not medical advice. "
    "Please consult a qualified clinician about your own situation."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and still have a dry cough and a "
    "mild fever. What should I know before I see a doctor?"
)


def build_messages(question: str, history=None) -> list:
    """Build the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A non-empty question is required.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages: list) -> dict:
    """Build the OpenAI-compatible request body for the local Spark server."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.3,
        "stream": False,
    }


def parse_response(data: dict) -> str:
    """Extract the answer and make sure the safety note is present."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "Malformed completion response: missing choices[0].message.content"
        ) from exc

    text = (content or "").strip()
    if not text:
        raise ValueError("Model returned an empty answer.")
    if SAFETY_NOTE not in text:
        text = text + "\n\n" + SAFETY_NOTE
    return text


def _http_post(url: str, body: bytes, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach the Spark-X2.6 endpoint at "
            f"{url}: {exc}. Start a local OpenAI-compatible server first "
            "(see README quickstart)."
        ) from exc


def ask(
    question: str,
    history=None,
    base_url: str = None,
    timeout: float = 120.0,
    transport=_http_post,
) -> str:
    """Send one question to Spark-X2.6-4B and return the plain-language answer."""
    root = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = root + "/chat/completions"
    payload = build_payload(build_messages(question, history))
    raw = transport(url, json.dumps(payload).encode("utf-8"), timeout)

    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Endpoint did not return valid JSON.") from exc
    return parse_response(data)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print(f"Q: {question}\n")
    try:
        answer = ask(question)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(f"A: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
