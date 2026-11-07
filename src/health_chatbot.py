#!/usr/bin/env python3
"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

A patient-education assistant: it explains health topics in plain language.
It does not diagnose, does not prescribe, and does not replace a clinician.

Run against a local OpenAI-compatible server (SGLang / vLLM / Ollama):

    python src/health_chatbot.py --base-url http://127.0.0.1:30000/v1
    python src/health_chatbot.py --offline   # print the exact prompt instead

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
DEFAULT_TIMEOUT = 120.0
MAX_QUESTION_CHARS = 2000

SYSTEM_PROMPT = (
    "You are a patient-education assistant powered by Spark-X2.6-4B. "
    "Explain health topics in plain, respectful language for readers with no medical training.\n"
    "Rules you must follow:\n"
    "1. Do not diagnose, do not tell the user which disease they have, and do not recommend "
    "prescription drugs or dosages.\n"
    "2. Explain what is commonly known, what usually happens next, and which questions are worth "
    "asking a clinician.\n"
    "3. Point the user to a qualified clinician or a local emergency service for any decision.\n"
    "4. Say clearly when information is uncertain or missing.\n"
    "5. Never claim the user's data was verified, and never claim the conversation is private "
    "beyond what this deployment actually guarantees.\n"
    "6. If the message suggests an emergency (for example chest pain, trouble breathing, heavy "
    "bleeding, fainting, or thoughts of self-harm), say so first and tell the user to contact "
    "emergency services now.\n"
    "Answer in the language the user wrote in."
)

DISCLAIMER = (
    "This is general health education, not medical advice, diagnosis, or treatment. "
    "Please consult a qualified clinician about your own situation."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago and since yesterday I have a dry cough, "
    "a mild fever, and I feel very tired. What should I pay attention to?"
)


def _clean_history(history):
    """Keep only well-formed user/assistant turns so the prompt stays bounded."""
    cleaned = []
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()[:MAX_QUESTION_CHARS]})
    return cleaned


def build_messages(question, history=None):
    """Return the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_clean_history(history))
    messages.append({"role": "user", "content": question.strip()[:MAX_QUESTION_CHARS]})
    return messages


def build_payload(question, history=None):
    """Build the OpenAI-compatible request body for Spark-X2.6-4B."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
    }


def extract_reply(body):
    """Parse the assistant text out of an OpenAI-compatible response body."""
    try:
        parsed = json.loads(body)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Spark endpoint returned a non-JSON body") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Spark endpoint returned an unexpected JSON shape")

    choices = parsed.get("choices") or []
    if not choices:
        raise RuntimeError("Spark endpoint returned no choices")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Spark endpoint returned an empty message")
    return content.strip()


def ask(question, base_url=None, history=None, timeout=DEFAULT_TIMEOUT):
    """Send the question to a local Spark-X2.6-4B OpenAI-compatible endpoint.

    Returns the assistant reply. Failures raise RuntimeError with an
    actionable message so the caller can show it to the user.
    """
    base = (base_url or DEFAULT_BASE_URL).rstrip("/")
    payload = build_payload(question, history)
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError("Spark endpoint returned HTTP %s: %s" % (exc.code, detail)) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            "Could not reach the Spark-X2.6-4B endpoint at %s. Start a local "
            "OpenAI-compatible server (see FAQ.md) or pass --base-url." % base
        ) from exc
    return extract_reply(body)


def render(reply):
    """Reply plus the fixed disclaimer shown to every user."""
    return "%s\n\n---\n%s" % (reply, DISCLAIMER)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health-information chatbot")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="print the prompt payload instead of calling the model",
    )
    args = parser.parse_args(argv)

    if args.offline:
        print(json.dumps(build_payload(args.question), indent=2, ensure_ascii=False))
        return 0

    try:
        reply = ask(args.question, base_url=args.base_url)
    except (RuntimeError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1

    print(render(reply))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
