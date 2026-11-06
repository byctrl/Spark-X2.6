"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Purpose: give plain-language general health education to people who cannot
easily reach a clinician, and route them to professional care when needed.

The model is only called for text generation; all safety rules live in the
system prompt and in the application code around the model boundary.
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
DEFAULT_TIMEOUT = 60

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-health help desk.\n"
    "Rules:\n"
    "1. Use plain language at roughly a 6th-grade reading level, short sentences.\n"
    "2. Give general health information only. Do NOT diagnose, do not name a "
    "presumed condition for this person, and do not recommend prescription "
    "drugs or doses.\n"
    "3. Always state which symptoms or situations need urgent in-person care, "
    "and encourage the person to see a qualified clinician about their own case.\n"
    "4. Never ask for or repeat identifying details (name, address, ID numbers).\n"
    "5. If the question is not general health education, say so and point to a "
    "better source.\n"
    "6. Answer in the language the user wrote in."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago. Now I have a cough and a fever of "
    "38.5C that will not go away. What should I do?"
)


class ModelError(RuntimeError):
    """Raised when the Spark-X2.6 endpoint cannot produce a usable answer."""


def _clean_question(question: str) -> str:
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must not be empty")
    return cleaned[:MAX_QUESTION_CHARS]


def build_messages(question: str) -> list:
    """Build the chat messages sent to Spark-X2.6-4B."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _clean_question(question)},
    ]


def build_payload(question: str, model_id: str = MODEL_ID) -> dict:
    """Build an OpenAI-compatible /chat/completions payload."""
    return {
        "model": model_id,
        "messages": build_messages(question),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.3,
    }


def extract_answer(response: dict) -> str:
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelError("model response has no choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ModelError("model returned an empty answer")
    return content.strip()


def _http_post_json(url: str, payload: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ModelError("cannot reach Spark endpoint %s: %s" % (url, exc)) from exc
    except json.JSONDecodeError as exc:
        raise ModelError("endpoint returned invalid JSON") from exc


def chat(question: str, base_url: str = None, transport=None,
         timeout: int = DEFAULT_TIMEOUT) -> str:
    """Ask Spark-X2.6-4B a health-education question and return the answer text.

    ``transport`` defaults to a real HTTP POST against an OpenAI-compatible
    Spark endpoint; tests pass a fake callable instead.
    """
    base = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    payload = build_payload(question)
    if transport is None:
        response = _http_post_json(base + "/chat/completions", payload, timeout)
    else:
        response = transport(payload)
    return extract_answer(response)


def main(argv: list = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print("Model: %s" % MODEL_ID)
    print("Question: %s\n" % question)
    try:
        print(chat(question))
    except (ModelError, ValueError) as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1
    print("\nReminder: this is general health information, not a diagnosis.")
    print("Please see a qualified clinician about your own situation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
