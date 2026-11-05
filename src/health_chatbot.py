"""Minimal health-information chatbot built on Spark-X2.6-4B.

AI for Good demo (HER Hack-Astron #7): answers general patient-education
questions in plain language, never diagnoses or prescribes, and always
points the reader to a qualified clinician.

The model is served by any OpenAI-compatible local runtime (SGLang, vLLM,
llama.cpp server, Spark-MLX-LLM). Point SPARK_BASE_URL at it; this module
needs no third-party packages and no cloud account.
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
    "You are a patient-education assistant for people with limited access to "
    "healthcare information. Explain things in short, plain sentences and "
    "avoid jargon; if you must use a medical term, define it.\n"
    "Rules:\n"
    "1. Do not diagnose, do not prescribe, and do not give dosages.\n"
    "2. Do not claim certainty. Separate general knowledge from what needs a "
    "clinician's examination.\n"
    "3. Always state which symptoms mean the reader should seek urgent care "
    "now.\n"
    "4. End by recommending that the reader confirm with a qualified health "
    "worker.\n"
    "5. If the question is outside health education, say so briefly."
)

DISCLAIMER = (
    "This is general health education, not a diagnosis or a prescription. "
    "Please confirm with a qualified health worker."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have had a dry "
    "cough and a fever in the evening. What could it be and when should I see "
    "a doctor?"
)


def build_messages(question: str):
    """Return the chat messages sent to Spark-X2.6-4B for one question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def complete(
    messages,
    base_url=None,
    model=MODEL_ID,
    max_tokens=MAX_NEW_TOKENS,
    timeout=REQUEST_TIMEOUT,
):
    """Call a local OpenAI-compatible Spark-X2.6-4B endpoint."""
    base_url = (
        base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL
    ).rstrip("/")
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "could not reach Spark-X2.6-4B at %s: %s" % (base_url, exc)
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("model server returned a non-JSON response") from exc
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "model server returned an unexpected response shape"
        ) from exc


def answer(question: str, complete_fn=None) -> str:
    """Answer a health-education question and append the safety disclaimer."""
    messages = build_messages(question)
    complete_fn = complete_fn or complete
    reply = (complete_fn(messages) or "").strip()
    if not reply:
        raise RuntimeError("model returned an empty response")
    if DISCLAIMER not in reply:
        reply = reply + "\n\n" + DISCLAIMER
    return reply


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    try:
        print(answer(question))
    except (ValueError, RuntimeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
