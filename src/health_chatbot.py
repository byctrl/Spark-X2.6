#!/usr/bin/env python3
"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

Public-interest demo: an inclusive, plain-language health education assistant.
The model explains and points to clinicians; it never diagnoses or prescribes.

A local OpenAI-compatible server for XHToken/Spark-X2.6-4B is expected, e.g.

    python -m sglang.launch_server --model-path XHToken/Spark-X2.6-4B --port 30000

Only the Python standard library is used (no extra dependencies).
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
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 120
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = (
    "You are Spark Health Explain, an educational assistant that helps people "
    "understand general health information in plain language.\n"
    "Rules you must follow:\n"
    "1. Use short sentences and everyday words (roughly a 6th-grade reading level). "
    "Explain any medical term you use.\n"
    "2. Never diagnose a condition, never give a medicine dose, and never say what "
    "treatment to take. Say what the topic is about and what questions to ask a doctor.\n"
    "3. If the person mentions chest pain, trouble breathing, fainting, heavy bleeding, "
    "sudden weakness or speech problems, or thoughts of self-harm, tell them to seek "
    "emergency care or a crisis line right away, before anything else.\n"
    "4. Ask for missing context (age group, how long symptoms last, existing conditions) "
    "instead of guessing.\n"
    "5. Remind the reader that this is general education, not medical advice, and that "
    "a licensed clinician should be consulted about their own situation."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days, since I came back from a trip. "
    "What should I know before I see a doctor?"
)

SAFETY_FOOTER = (
    "This is general health education, not medical advice or a diagnosis. "
    "Please talk to a licensed clinician about your own situation."
)


class ModelResponseError(ValueError):
    """Raised when the model endpoint returns something unusable."""


def build_messages(question, history=None):
    """Build the chat messages: system prompt, optional history, then the question."""
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


def build_payload(messages, max_new_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body for Spark-X2.6-4B."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": int(max_new_tokens),
        "temperature": TEMPERATURE,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text and reject empty or malformed responses."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelResponseError(f"unexpected response shape: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise ModelResponseError("model returned empty content")
    return content.strip()


def _post_json(url, payload, timeout):
    """POST JSON with the standard library; returns the decoded response body."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def call_model(messages, base_url=DEFAULT_BASE_URL, timeout=REQUEST_TIMEOUT, poster=None):
    """Send the messages to the local Spark-X2.6-4B endpoint and return text."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = build_payload(messages)
    poster = poster or _post_json
    return parse_response(poster(url, payload, timeout))


def answer(question, history=None, base_url=DEFAULT_BASE_URL,
           timeout=REQUEST_TIMEOUT, poster=None):
    """Answer one health question and return a reviewable result dict."""
    messages = build_messages(question, history)
    text = call_model(messages, base_url=base_url, timeout=timeout, poster=poster)
    return {
        "model": MODEL_ID,
        "question": question.strip(),
        "messages": messages,
        "answer": text,
        "notice": SAFETY_FOOTER,
    }


def render(reply):
    """Format a reply for terminal output."""
    return f"{reply['answer']}\n\n---\n{reply['notice']}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a plain-language health question."
    )
    parser.add_argument("question", nargs="*", help="question text; defaults to the demo question")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--offline", action="store_true", help="print the request payload only")
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip() or DEMO_QUESTION
    try:
        if args.offline:
            payload = build_payload(build_messages(question))
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        reply = answer(question, base_url=args.base_url)
    except (urllib.error.URLError, ModelResponseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(reply, ensure_ascii=False, indent=2))
    else:
        print(render(reply))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
