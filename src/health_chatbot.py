"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

Plain-language patient education only: the assistant explains health topics,
never diagnoses, never recommends or adjusts medicines or doses, and always
points the reader to a qualified clinician.

The application talks to a locally hosted Spark-X2.6-4B OpenAI-compatible
endpoint (see FAQ.md for supported runtimes). No cloud service or account is
required. Point ``SPARK_BASE_URL`` at your server, for example
``http://127.0.0.1:30000/v1`` for a local SGLang/vLLM deployment.

Usage:
    python src/health_chatbot.py "what is a fever?"
    python src/health_chatbot.py            # runs DEMO_QUESTION
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who have limited access "
    "to health information. Explain health topics in plain language at about a "
    "6th-grade reading level, in short paragraphs, and define any medical term "
    "you have to use. Do not diagnose, do not recommend or change medicines or "
    "doses, and do not replace a clinician. If the question suggests an "
    "emergency (for example chest pain, trouble breathing, fainting, heavy "
    "bleeding, or thoughts of self-harm), say clearly that the person should "
    "contact local emergency services right now. If you are unsure, say so "
    "instead of guessing. End every answer with exactly one line: "
    "'This is general education, not medical advice - please see a qualified "
    "clinician about your own situation.'"
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a cough and a "
    "fever that comes and goes. What could be going on, and what should I do next?"
)


def build_messages(question: str, system_prompt: str = SYSTEM_PROMPT):
    """Build the chat messages for a single user question."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def build_payload(
    question: str,
    model: str = MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """Build the OpenAI-compatible request body."""
    return {
        "model": model,
        "messages": build_messages(question),
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "stream": False,
    }


def extract_reply(body: str) -> str:
    """Pull the assistant text out of an OpenAI-compatible response body."""
    try:
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("unexpected response from Spark-X2.6-4B endpoint") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Spark-X2.6-4B returned an empty answer")
    return content.strip()


def chat(
    question: str,
    base_url: str | None = None,
    payload: dict | None = None,
    timeout: int = REQUEST_TIMEOUT,
    opener=None,
) -> str:
    """Send one question to the local Spark-X2.6-4B server and return the answer."""
    root = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = root + "/chat/completions"
    body = json.dumps(payload or build_payload(question)).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "could not reach Spark-X2.6-4B at {} (is the local server running?): {}".format(
                url, exc
            )
        ) from exc
    return extract_reply(raw)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a plain-language health question."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="health question; defaults to the built-in DEMO_QUESTION",
    )
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip() or DEMO_QUESTION
    print(chat(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
