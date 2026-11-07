"""Minimal Spark-X2.6-4B health-information chatbot.

A small, dependency-free demo for the HER Hack-Astron #7 AI for Good
challenge: an inclusive plain-language health knowledge assistant built on
XHToken/Spark-X2.6-4B.

The chatbot only explains general health information. It does not diagnose,
prescribe, or replace a clinician, and it tells the user to seek professional
care when a situation may be urgent.

Run it against a local OpenAI-compatible server::

    python src/health_chatbot.py "batuk dan demam setelah perjalanan"

Environment variables:
    SPARK_BASE_URL   default http://127.0.0.1:30000/v1
    SPARK_API_KEY    default "EMPTY" (most local servers ignore it)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who may have limited "
    "access to healthcare. Explain general health information in simple, "
    "respectful language (roughly a 6th-grade reading level).\n"
    "Rules:\n"
    "1. Never diagnose, never name a definitive disease, and never prescribe "
    "or dose medication.\n"
    "2. Never claim to replace a doctor, nurse, or pharmacist.\n"
    "3. Say clearly which symptoms or situations need urgent professional "
    "care (for example trouble breathing, chest pain, fainting, heavy "
    "bleeding, confusion, or a fever in a very young infant).\n"
    "4. Mention that local emergency numbers and clinics vary, and encourage "
    "the user to contact a local health worker.\n"
    "5. If the question is unclear or outside health, say so briefly and ask "
    "one short clarifying question.\n"
    "6. Answer in the language the user writes in.\n"
    "End every answer with: \"This is general information, not a diagnosis. "
    "Please see a health worker for your own situation.\""
)

# Demo scenario: cough and fever after travelling.
DEMO_QUESTION = (
    "Saya batuk kering dan demam 38 derajat Celsius sejak tiga hari lalu, "
    "setelah pulang dari perjalanan. Apa yang sebaiknya saya lakukan?"
)


def build_messages(question, history=None):
    """Return the OpenAI-style chat messages for one user question."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": text})
    return messages


def build_payload(messages, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS, temperature=0.2):
    """Build the request body sent to the OpenAI-compatible endpoint."""
    return {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text from a chat-completions response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response from model endpoint: %r" % (data,)) from exc

    text = (content or "").strip()
    if not text:
        raise ValueError("model returned an empty answer")
    return text


def ask(question, history=None, base_url=None, api_key=None, timeout=DEFAULT_TIMEOUT):
    """Send one question to a local Spark-X2.6-4B server and return the answer."""
    root = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = root + "/chat/completions"
    key = api_key or os.environ.get("SPARK_API_KEY") or "EMPTY"

    body = json.dumps(build_payload(build_messages(question, history))).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "cannot reach %s (%s). Start a local Spark-X2.6-4B server first."
            % (url, exc)
        ) from exc

    return parse_response(data)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION

    try:
        answer = ask(question)
    except (ValueError, RuntimeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1

    print("Q:", question)
    print()
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
