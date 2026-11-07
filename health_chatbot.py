"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

A small, safety-bounded application for inclusive health education. It sends a
bounded prompt to a locally hosted OpenAI-compatible Spark-X2.6-4B endpoint and
returns a plain-language answer. It never diagnoses, never prescribes, and
always points the reader to a qualified clinician.

Usage:
    python health_chatbot.py --offline            # show the exact prompt/payload
    python health_chatbot.py --question "..."     # call the local endpoint

The endpoint defaults to http://127.0.0.1:30000/v1 and can be overridden with
the SPARK_BASE_URL environment variable or --base-url.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
BASE_URL_ENV = "SPARK_BASE_URL"
TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant powered by Spark-X2.6-4B. "
    "Explain general health topics in simple, everyday language that a person "
    "with low health literacy can understand. Use short sentences and define "
    "any medical term you use.\n"
    "Hard rules:\n"
    "1. Do not diagnose, do not name a likely disease as the user's condition, "
    "and do not recommend prescription drugs, dosages, or treatments.\n"
    "2. Do not claim to replace a doctor, nurse, or pharmacist.\n"
    "3. Always tell the user which kind of clinician or service to contact and "
    "which warning signs mean they should seek urgent care now.\n"
    "4. If the user describes an emergency (for example chest pain, trouble "
    "breathing, heavy bleeding, fainting, or thoughts of self-harm), start the "
    "answer by telling them to contact local emergency services immediately.\n"
    "5. If information is missing, ask one short clarifying question instead of "
    "guessing.\n"
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days, and I got back from a "
    "trip last week. What should I know before I see a doctor?"
)

SAFETY_NOTICE = (
    "This is general health education, not medical advice, diagnosis, or "
    "treatment. Please contact a qualified clinician about your own situation. "
    "If you have severe symptoms, contact your local emergency number now."
)


def build_messages(question, history=None):
    """Return the bounded chat messages sent to Spark-X2.6-4B."""
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


def build_payload(question, history=None, max_tokens=MAX_NEW_TOKENS, temperature=0.3):
    """Build the OpenAI-compatible request body for the local Spark endpoint."""
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "stream": False,
    }


def extract_answer(response):
    """Pull the final answer text out of a chat-completion response."""
    try:
        choice = response["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("response does not contain choices[0]") from exc
    message = choice.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def with_safety_notice(answer):
    """Append the fixed safety notice so the UI can never drop it."""
    text = (answer or "").strip()
    if not text:
        raise ValueError("answer must be a non-empty string")
    if SAFETY_NOTICE in text:
        return text
    return text + "\n\n" + SAFETY_NOTICE


def endpoint_url(base_url=None):
    root = (base_url or os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")
    return root + "/chat/completions"


def chat(question, history=None, base_url=None, max_tokens=MAX_NEW_TOKENS,
         temperature=0.3, timeout=TIMEOUT_SECONDS):
    """Ask Spark-X2.6-4B for a plain-language health answer."""
    url = endpoint_url(base_url)
    body = json.dumps(build_payload(question, history, max_tokens, temperature)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "could not reach the Spark endpoint at %s (%s); start a local "
            "OpenAI-compatible server for %s and retry" % (url, exc, MODEL_ID)
        ) from exc
    return with_safety_notice(extract_answer(payload))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health-information chatbot")
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--offline", action="store_true",
                        help="print the demo payload without calling the model")
    args = parser.parse_args(argv)

    if args.offline:
        payload = build_payload(args.question, max_tokens=args.max_tokens)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(chat(args.question, base_url=args.base_url, max_tokens=args.max_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
