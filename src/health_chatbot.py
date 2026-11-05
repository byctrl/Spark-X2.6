#!/usr/bin/env python3
"""Minimal public-interest health-education chatbot powered by Spark-X2.6-4B.

The application answers plain-language health questions for people with limited
access to care. It only produces patient education: it does not diagnose,
prescribe, or replace a clinician. Spark-X2.6-4B is a text-only model, so the
chatbot passes text to a local OpenAI-compatible endpoint (SGLang / vLLM / MLX)
and never claims that model output is medical ground truth.

Run a demo request without a server:

    python src/health_chatbot.py --demo --dry-run

Run against a local Spark-X2.6-4B server:

    python src/health_chatbot.py --demo --base-url http://127.0.0.1:30000/v1
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
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
DEFAULT_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public health helpline. "
    "Explain health information in plain, respectful language for readers with "
    "limited literacy or limited access to care.\n"
    "Rules:\n"
    "1. Do not diagnose, prescribe, or recommend specific medicines or doses.\n"
    "2. Do not claim that this answer replaces a clinician.\n"
    "3. Encourage the reader to contact a qualified clinician or a local "
    "emergency service when symptoms are severe, worsening, or urgent.\n"
    "4. Do not ask for or repeat personal identifiers.\n"
    "5. If the information is insufficient, say so and list what a clinician "
    "would need to know.\n"
    "6. Answer in the language of the question, using short sentences and "
    "bullet points."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and now I have a cough and fever. "
    "What should I know before I see a doctor?"
)


def build_messages(question, history=None):
    """Build the chat messages for one turn, keeping the safety prompt first."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_payload(messages, model_id=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
    """Build an OpenAI-compatible chat-completions payload."""
    return {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant answer and reject empty or malformed completions."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected completion payload") from exc

    content = (content or "").strip()
    if not content:
        raise ValueError("model returned an empty answer")
    return content


def chat(question, base_url=DEFAULT_BASE_URL, api_key=None, history=None, timeout=DEFAULT_TIMEOUT):
    """Send one question to a locally hosted Spark-X2.6-4B endpoint."""
    payload = build_payload(build_messages(question, history))
    request = urllib.request.Request(
        url=base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("Authorization", "Bearer " + api_key)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "could not reach the Spark-X2.6-4B endpoint at " + base_url + ": " + str(exc)
        ) from exc

    try:
        data = json.loads(body)
    except ValueError as exc:
        raise RuntimeError("endpoint did not return JSON") from exc
    return parse_response(data)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-education chatbot (patient education only)."
    )
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=os.environ.get("SPARK_API_KEY"), help="optional bearer token")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="request timeout in seconds")
    parser.add_argument("--demo", action="store_true", help="use the bundled demo question")
    parser.add_argument("--dry-run", action="store_true", help="print the request payload and exit")
    args = parser.parse_args(argv)

    question = DEMO_QUESTION if args.demo else args.question

    if args.dry_run:
        payload = build_payload(build_messages(question))
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    try:
        answer = chat(
            question,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
        )
    except (RuntimeError, ValueError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1

    print(answer)
    print("\nReminder: this is general education, not a diagnosis. Please consult a clinician.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
