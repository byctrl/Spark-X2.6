"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

This module is a small, dependency-free example application for the
HER Hack-Astron #7 challenge. It sends a bounded, patient-education prompt
to a locally hosted Spark-X2.6-4B model exposed through an
OpenAI-compatible endpoint and returns plain-language health information.

The model never makes a diagnosis or prescribes treatment: it explains,
in plain language, and points people to a qualified clinician.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
REQUEST_TIMEOUT = float(os.environ.get("SPARK_TIMEOUT", "120"))

DISCLAIMER = (
    "Educational information only - not a diagnosis or a prescription. "
    "Please consult a qualified clinician about your own situation."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health topics in plain, "
    "respectful language that a person without medical training can follow. "
    "You do not diagnose, you do not prescribe, and you never tell the user "
    "to stop or change a prescribed treatment. State clearly which symptoms "
    "or situations need urgent in-person care, and always recommend "
    "consulting a qualified clinician. If the question is outside health "
    "education, say so briefly and suggest an appropriate source of help. "
    "Keep answers short and structured, and never invent dosages, test "
    "results, or statistics."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a cough and "
    "some fever in the evenings. What should I pay attention to, and when "
    "should I see a doctor?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B for one question."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS):
    """Build an OpenAI-compatible chat-completions payload."""
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }


def post_json(url, payload, timeout=REQUEST_TIMEOUT):
    """POST a JSON payload and return the decoded JSON response."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(
            "Spark endpoint returned HTTP {0}: {1}".format(exc.code, detail)
        ) from exc


def extract_answer(response):
    """Validate the endpoint response and return the assistant text."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected model response: no choices/message/content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def ask(
    question,
    base_url=DEFAULT_BASE_URL,
    history=None,
    model=MODEL_ID,
    max_tokens=MAX_NEW_TOKENS,
    _post=None,
):
    """Ask Spark-X2.6-4B one health-education question.

    ``base_url`` points at a local SGLang / vLLM / llama.cpp OpenAI-compatible
    server serving the Spark-X2.6-4B weights.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")

    poster = _post or post_json
    payload = build_payload(
        build_messages(question, history), model=model, max_tokens=max_tokens
    )
    url = base_url.rstrip("/") + "/chat/completions"
    return extract_answer(poster(url, payload))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Minimal Spark-X2.6-4B health-information chatbot"
    )
    parser.add_argument(
        "question", nargs="*", help="question to ask; omit to use the demo question"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible Spark endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--demo", action="store_true", help="run the built-in demo question"
    )
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip()
    if args.demo or not question:
        question = DEMO_QUESTION

    print("Q: " + question)
    print("A: " + ask(question, base_url=args.base_url))
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
