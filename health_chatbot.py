"""Spark-X2.6-4B health-education chatbot (HER Hack-Astron #7 entry).

A small, offline-friendly chatbot that explains general health topics in plain
language for people with limited access to health information. It is an
*education* tool, not a diagnosis tool: the system prompt forbids diagnosis and
prescription, and every answer carries a fixed disclaimer.

The application talks to Spark-X2.6-4B through its documented
OpenAI-compatible endpoint, so it runs fully locally with no cloud account:

    python -m sglang.launch_server --model-path XHToken/Spark-X2.6-4B --port 30000
    python health_chatbot.py
    python health_chatbot.py --base-url http://127.0.0.1:30000/v1

Set SPARK_BASE_URL to override the default endpoint URL.

Demo scenario (DEMO_QUESTION): a dry cough with mild fever after travel.
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
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 180

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain general health topics in "
    "simple, respectful language for a reader with no medical training. "
    "Rules: never diagnose, never prescribe, never give a medicine dose, and "
    "never claim certainty about an individual's condition. Say clearly when a "
    "symptom needs urgent professional care, and always remind the reader to "
    "consult a qualified clinician about their own situation. If the question "
    "is outside general health education, say so briefly and suggest who to "
    "ask instead. Keep the answer under 250 words."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak tiga hari lalu, dan baru pulang "
    "dari perjalanan. Apa yang sebaiknya saya perhatikan?"
)

DISCLAIMER = (
    "This is general health information, not medical advice or a diagnosis. "
    "Please consult a qualified clinician about your own situation."
)


def build_messages(question, history=None):
    """Return the chat messages sent to Spark-X2.6-4B for one question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def parse_reply(raw):
    """Extract the assistant text from an OpenAI-compatible chat response."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("endpoint returned a non-JSON response") from exc
    else:
        data = raw

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "endpoint response is missing choices[0].message.content"
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("model returned an empty reply")
    return content.strip()


def _post_json(url, payload, timeout=REQUEST_TIMEOUT):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def chat(question, history=None, base_url=None, transport=None, max_new_tokens=MAX_NEW_TOKENS):
    """Send one question to Spark-X2.6-4B and return the raw assistant text."""
    messages = build_messages(question, history)
    url = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
    }

    sender = transport or _post_json
    try:
        raw = sender(url + "/chat/completions", payload)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            "could not reach a Spark-X2.6-4B OpenAI-compatible endpoint at "
            + url
            + "; start a local server or set SPARK_BASE_URL"
        ) from exc

    return parse_reply(raw)


def answer(question, history=None, **kwargs):
    """Return a plain-language answer with the fixed education disclaimer."""
    reply = chat(question, history, **kwargs)
    if DISCLAIMER not in reply:
        reply = reply + "\n\n" + DISCLAIMER
    return reply


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a general health-education question."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="question to ask; omit to run the built-in demo scenario",
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip() or DEMO_QUESTION
    endpoint = args.base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL
    print("[model] " + MODEL_ID + " via " + endpoint)
    print("[question] " + question)
    print()

    try:
        print(answer(question, base_url=args.base_url, max_new_tokens=args.max_new_tokens))
    except (RuntimeError, ValueError) as exc:
        print("[error] " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
