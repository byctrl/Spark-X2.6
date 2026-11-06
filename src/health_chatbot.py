"""Minimal health-information chatbot demo for Spark-X2.6-4B.

A safety-bounded patient-education assistant: it explains health topics in
plain language and never diagnoses, prescribes, or replaces a clinician.

Run against a local OpenAI-compatible server:

    python src/health_chatbot.py
    python src/health_chatbot.py --base-url http://127.0.0.1:30000/v1 --question "..."
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
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-health access project. "
    "Explain health topics in plain language at roughly a 6th-grade reading level, "
    "in the same language the user writes in.\n"
    "Rules:\n"
    "1. Do NOT diagnose, do NOT name a probable condition for this person, and do NOT "
    "recommend prescription medicines or dosages.\n"
    "2. Do NOT claim to replace a clinician. Always advise seeing a qualified health "
    "professional for personal decisions.\n"
    "3. State clearly when something is an emergency (for example trouble breathing, "
    "chest pain, fainting, or heavy bleeding) and tell the user to seek urgent care or "
    "call local emergency services.\n"
    "4. Ask at most one brief clarifying question if key context is missing; otherwise "
    "answer directly.\n"
    "5. If you are unsure, say so. Never invent sources, statistics, or guidelines.\n"
    "6. Keep the answer short: a few short paragraphs or a short bullet list."
)

DEMO_QUESTION = (
    "I got back from a two-week trip three days ago and now I have a dry cough and "
    "a mild fever. What should I know before I talk to a doctor?"
)


def build_messages(question, history=None):
    """Build the chat-completions message list for one user question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question, history=None, max_new_tokens=MAX_NEW_TOKENS):
    """Build the request body sent to the OpenAI-compatible endpoint."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": int(max_new_tokens),
        "temperature": 0.2,
        "stream": False,
    }


def extract_reply(response):
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response shape from model server") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty reply")
    return content.strip()


def ask(question, history=None, base_url=DEFAULT_BASE_URL, max_new_tokens=MAX_NEW_TOKENS):
    """Send one question to a locally hosted Spark-X2.6-4B and return its reply."""
    payload = json.dumps(
        build_payload(question, history, max_new_tokens), ensure_ascii=False
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        body = json.loads(response.read().decode("utf-8"))
    return extract_reply(body)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot demo"
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible endpoint"
    )
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
    args = parser.parse_args(argv)

    print("model:", MODEL_ID)
    print("question:", args.question)
    print("-" * 60)
    try:
        print(ask(args.question, base_url=args.base_url))
    except (urllib.error.URLError, ValueError) as exc:
        print("Could not reach the local model server:", exc)
        print("Start Spark-X2.6-4B behind an OpenAI-compatible server and retry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
