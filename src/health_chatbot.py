"""Minimal health-information chatbot powered by Spark-X2.6-4B.

Public-interest demo for HER Hack-Astron #7: plain-language health education
for people who cannot easily reach a clinician. The model gives general
education only - it never diagnoses and never prescribes.

The chatbot talks to any OpenAI-compatible endpoint (vLLM, SGLang, llama.cpp,
Spark-MLX-LLM, Ollama). No third-party dependencies.

Usage:
    python src/health_chatbot.py --question "..."

Environment:
    SPARK_BASE_URL   default http://127.0.0.1:30000/v1
    SPARK_API_KEY    optional bearer token
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
EMERGENCY_PREFIX = "EMERGENCY:"
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = (
    "You are a patient-education assistant at a community health desk. "
    "Answer in plain language at roughly a 6th-grade reading level, in the "
    "same language as the user. Give general health information, explain "
    "common terms, and list questions the person could ask a clinician.\n"
    "Hard rules:\n"
    "1. Never diagnose, never name a probable disease as a conclusion, and "
    "never prescribe or give a dose of any medicine.\n"
    "2. Never tell the user to delay or stop professional care.\n"
    "3. If the message suggests an emergency (chest pain, trouble breathing, "
    "heavy bleeding, fainting, thoughts of self-harm, a very sick infant), "
    "start the reply with 'EMERGENCY:' and tell them to contact local "
    "emergency services now.\n"
    "4. State clearly that this is general information, not medical advice, "
    "and recommend a qualified clinician for anything personal.\n"
    "5. Do not ask for names, ID numbers, addresses or other personal data."
)

DEMO_QUESTION = (
    "Saya demam tiga hari dan batuk kering sejak pulang dari luar kota. "
    "Apa yang sebaiknya saya tanyakan ke dokter saat periksa nanti?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("history turns must be dicts")
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
    """Build an OpenAI-compatible chat-completions payload."""
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("messages must end with a user turn")
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": int(max_new_tokens),
        "temperature": TEMPERATURE,
        "stream": False,
    }


def extract_reply(body):
    """Pull the assistant text out of an OpenAI-compatible response body."""
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("unexpected response shape: %r" % (body,)) from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("model returned an empty reply")
    return content.strip()


def generate(messages, base_url=DEFAULT_BASE_URL, api_key=None,
             timeout=REQUEST_TIMEOUT, opener=None):
    """Send the messages to a local Spark-X2.6-4B endpoint and return the reply."""
    payload = build_payload(messages)
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    token = api_key or os.environ.get("SPARK_API_KEY")
    if token:
        headers["Authorization"] = "Bearer " + token

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "cannot reach a Spark-X2.6 endpoint at %s: %s" % (url, exc)
        ) from exc
    except ValueError as exc:
        raise RuntimeError("endpoint returned invalid JSON") from exc
    return extract_reply(body)


def is_emergency_reply(reply):
    """True when the model flagged the message as needing emergency care."""
    return reply.strip().upper().startswith(EMERGENCY_PREFIX)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot (education only)"
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args(argv)

    print("model: %s" % MODEL_ID)
    print("endpoint: %s" % args.base_url)
    print("question: %s" % args.question)
    print("")

    try:
        reply = generate(build_messages(args.question), base_url=args.base_url)
    except RuntimeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        print("start a local OpenAI-compatible Spark-X2.6 server first "
              "(see README.md#quickstart).", file=sys.stderr)
        return 1

    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
