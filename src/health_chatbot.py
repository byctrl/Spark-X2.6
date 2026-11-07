#!/usr/bin/env python3
"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Goal: an inclusive, plain-language health knowledge assistant for people who
otherwise have no easy access to reliable general health information.

Standard library only. The script talks to a locally hosted, OpenAI-compatible
Spark-X2.6-4B endpoint (SGLang / vLLM / llama.cpp server / Ollama ``/v1``).

    export SPARK_BASE_URL=http://127.0.0.1:30000/v1
    python src/health_chatbot.py --question "I have had a cough and fever..."

Safety boundary: the model only explains health topics. It never diagnoses,
prescribes, or replaces a clinician; a human health worker remains the
decision maker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
TOP_P = 0.9
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant inside a public-interest health "
    "information service built on Spark-X2.6-4B.\n"
    "Rules:\n"
    "1. Explain in plain language: short sentences, no unexplained jargon.\n"
    "2. Never diagnose, never name a medicine or a dose, and never tell the "
    "user to start, stop or change a treatment.\n"
    "3. Never claim to replace a clinician. Always advise seeing a qualified "
    "health worker for personal decisions.\n"
    "4. If the message suggests an emergency (for example chest pain, "
    "difficulty breathing, heavy bleeding, fainting, a seizure, or thoughts "
    "of self-harm), say clearly that emergency care is needed now and to "
    "contact local emergency services.\n"
    "5. Ask at most one short clarifying question when key information is "
    "missing.\n"
    "6. Do not invent sources, statistics, phone numbers or organisation "
    "names. Say so when you are unsure.\n"
    "7. Reply in the language the user writes in.\n"
    "Output format: three short sections - 'What this may mean', 'What you "
    "can do now', 'When to get help' - followed by a one-line reminder that "
    "this is general education, not medical advice."
)

DEMO_QUESTION = (
    "I came back from a two-week trip ten days ago. Since three days ago I "
    "have a dry cough, a fever of about 38.5 C in the evening, and I feel "
    "very tired. I have no chest pain and I can breathe normally. What could "
    "this be and what should I do next?"
)


def build_messages(question, history=None):
    """Build the chat messages: system prompt, optional history, user turn."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_payload(messages, model=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body for Spark-X2.6-4B."""
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": False,
    }


def extract_answer(response):
    """Pull the assistant text out of a chat-completions response."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "unexpected response from the Spark-X2.6 endpoint: %r" % (response,)
        ) from exc
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError("Spark-X2.6 returned an empty answer")
    return content


def _default_transport(url, payload, timeout):
    """POST JSON with urllib and return the decoded JSON response."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def chat(
    question,
    history=None,
    base_url=None,
    model=MODEL_ID,
    transport=None,
    max_new_tokens=MAX_NEW_TOKENS,
):
    """Ask Spark-X2.6-4B one health-information question and return the answer."""
    messages = build_messages(question, history)
    payload = build_payload(messages, model=model, max_new_tokens=max_new_tokens)
    base = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = base + "/chat/completions"
    send = transport or _default_transport
    return extract_answer(send(url, payload, REQUEST_TIMEOUT))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a general health-information question."
    )
    parser.add_argument("--question", default=DEMO_QUESTION, help="user question")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (default: $SPARK_BASE_URL or %s)"
        % DEFAULT_BASE_URL,
    )
    parser.add_argument("--model", default=MODEL_ID, help="served model name")
    args = parser.parse_args(argv)

    try:
        answer = chat(args.question, base_url=args.base_url, model=args.model)
    except Exception as exc:  # network, server or parsing failure
        print("error: %s" % exc, file=sys.stderr)
        return 1

    print("Q: %s" % args.question)
    print()
    print(answer)
    print()
    print(
        "Reminder: this is general health education, not medical advice. "
        "Please consult a qualified health worker for personal decisions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
