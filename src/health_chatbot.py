"""Minimal health-information chatbot demo built on Spark-X2.6-4B.

This is the smallest working artifact for the HER Hack-Astron #7
"AI for Good" challenge. It turns a user question into a bounded
patient-education prompt, sends it to a locally hosted
Spark-X2.6-4B OpenAI-compatible endpoint, and returns a reviewable
plain-language answer.

The model is an education aid, not a clinician: it is instructed not
to diagnose or prescribe, and every reply carries a disclaimer that
points the reader to a qualified professional.

Run (after starting a local Spark-X2.6-4B server, e.g. vLLM/SGLang):

    python src/health_chatbot.py
    python src/health_chatbot.py --question "How do I store insulin?"
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_ENDPOINT = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = (
    "You are a patient-education assistant helping people understand general "
    "health information.\n"
    "Rules you must follow:\n"
    "1. Use short sentences and plain language (about a 6th-grade reading level).\n"
    "2. Never diagnose, never name a likely disease, never prescribe or dose a "
    "medicine, and never tell the reader to stop a prescribed treatment.\n"
    "3. Explain what is generally known, what the reader can safely observe, and "
    "which warning signs mean they should seek care urgently.\n"
    "4. Always end by recommending a qualified clinician, pharmacist or local "
    "emergency service for personal decisions.\n"
    "5. If the question is outside general health education, say so briefly and "
    "suggest who can help.\n"
    "6. Do not ask for or repeat identifying details such as names, addresses or "
    "ID numbers."
)

DEMO_QUESTION = (
    "I came back from a two-week trip ten days ago and now I have a dry cough "
    "and a mild fever in the evening. What should I pay attention to, and when "
    "should I see a doctor?"
)

DISCLAIMER = (
    "This is general health education, not medical advice, diagnosis or "
    "treatment. Please talk to a qualified clinician or pharmacist about your "
    "own situation, and contact local emergency services if symptoms are severe "
    "or getting worse."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    ``history`` is an optional list of ``{"role", "content"}`` turns from the
    previous conversation; only ``user`` and ``assistant`` turns with non-empty
    content are kept so the prompt stays bounded.
    """
    text = (question or "").strip()
    if not text:
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})
    return messages


def build_payload(messages, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body."""
    if not messages:
        raise ValueError("messages must not be empty")
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }


def chat_completions_url(endpoint=DEFAULT_ENDPOINT):
    return (endpoint or "").rstrip("/") + "/chat/completions"


def parse_response(body):
    """Extract the answer text from an OpenAI-compatible response body."""
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("malformed model response") from exc
    text = (content or "").strip()
    if not text:
        raise ValueError("model returned an empty answer")
    return format_reply(text)


def format_reply(text):
    """Append the safety disclaimer when the model did not include it."""
    text = (text or "").strip()
    if not text:
        raise ValueError("reply must not be empty")
    if DISCLAIMER in text:
        return text
    return text + "\n\n" + DISCLAIMER


def ask(question, history=None, endpoint=DEFAULT_ENDPOINT, timeout=120, opener=None):
    """Send one question to a locally hosted Spark-X2.6-4B endpoint."""
    opener = opener or urllib.request.urlopen
    payload = build_payload(build_messages(question, history))
    request = urllib.request.Request(
        chat_completions_url(endpoint),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach the Spark-X2.6-4B endpoint at {0}. "
            "Start a local OpenAI-compatible server first.".format(
                chat_completions_url(endpoint)
            )
        ) from exc
    try:
        body = json.loads(raw)
    except ValueError as exc:
        raise ValueError("endpoint did not return JSON") from exc
    return parse_response(body)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="OpenAI-compatible base URL serving " + MODEL_ID,
    )
    args = parser.parse_args(argv)

    print("Model: " + MODEL_ID)
    print("Question: " + args.question.strip())
    print("-" * 60)
    print(ask(args.question, endpoint=args.endpoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
