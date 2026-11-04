"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

A small, dependency-free demo for the Spark-X2.6 Open Source AI for Good
Challenge: a plain-language patient-education assistant that talks to a
locally served Spark-X2.6-4B OpenAI-compatible endpoint.

Safety boundary: this is education only. It does not diagnose, does not
recommend prescription medicines or dosages, and always points the reader to
a qualified clinician or local health service for their own situation.

Usage:
    python src/health_chatbot.py --question "..."
    python src/health_chatbot.py            # runs DEMO_QUESTION

The endpoint defaults to http://127.0.0.1:30000/v1 (SGLang/vLLM style
OpenAI-compatible server) and can be changed with --base-url.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

# Shown to the reader together with every answer.
DISCLAIMER = (
    "This is general health information, not medical advice. "
    "Please contact a qualified clinician or your local health service "
    "about your own situation."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant. You help people understand "
    "general health information in plain, everyday language. You are not a "
    "clinician and you do not provide medical care.\n\n"
    "Follow these rules:\n"
    "1. Use short sentences and common words. Explain any medical term you "
    "must use.\n"
    "2. Never diagnose, never present a likely disease as a conclusion, and "
    "never suggest prescription medicines, dosages, or treatments.\n"
    "3. Share only general, widely accepted information. If you are unsure, "
    "say so plainly.\n"
    "4. Name the warning signs that mean the reader should seek urgent care, "
    "and remind them to contact a qualified clinician or local health "
    "service for their own situation.\n"
    "5. Never ask for or repeat names, addresses, ID numbers, or other "
    "personal data.\n"
    "6. If the question is not about health, say briefly that you only cover "
    "health information and suggest a better source.\n"
    "7. Reply in the language of the question. Keep the answer under 250 "
    "words.\n"
)

# Demo scenario: cough and fever after travel.
DEMO_QUESTION = (
    "I came back from a trip five days ago and now I have a cough and a "
    "fever of 38.4 C. What should I pay attention to?"
)

VALID_HISTORY_ROLES = ("user", "assistant")


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt always comes first; optional earlier turns follow it;
    the current question is last.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("each history turn must be a dict")
        role = turn.get("role")
        content = turn.get("content")
        if role not in VALID_HISTORY_ROLES or not isinstance(content, str) or not content.strip():
            raise ValueError("history turns need role 'user'/'assistant' and non-empty content")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question, history=None, max_tokens=MAX_NEW_TOKENS, temperature=0.2):
    """Build the OpenAI-compatible request body for one chat completion."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text from an OpenAI-compatible response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected response shape from the Spark-X2.6 endpoint")

    text = (content or "").strip()
    if not text:
        raise ValueError("the model returned an empty answer")
    return text


def ask(question, base_url=DEFAULT_BASE_URL, history=None, max_tokens=MAX_NEW_TOKENS, timeout=REQUEST_TIMEOUT):
    """Send one question to a local Spark-X2.6-4B endpoint and return the text."""
    payload = json.dumps(build_payload(question, history, max_tokens)).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return parse_response(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health-information chatbot")
    parser.add_argument("--question", default=DEMO_QUESTION, help="health question to ask")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible endpoint base URL")
    args = parser.parse_args(argv)

    print("Model:", MODEL_ID)
    print("Question:", args.question)
    print("-" * 60)
    try:
        answer = ask(args.question, base_url=args.base_url)
    except Exception as exc:  # noqa: BLE001 - demo CLI should not traceback
        print("Could not get an answer:", exc)
        print("Start a local Spark-X2.6-4B OpenAI-compatible server and retry.")
        return 1
    print(answer)
    print("-" * 60)
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
