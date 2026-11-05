"""Minimal health-information chatbot built on Spark-X2.6-4B.

Demo entry for HER Hack-Astron #7 (AI for Good): a plain-language patient
education assistant. The model is used only to explain health topics; it must
never diagnose or prescribe, and it always points the user to a clinician.

The script talks to a locally served Spark-X2.6-4B through an
OpenAI-compatible endpoint, so it stays dependency-free (stdlib only).

Run the demo:
    python src/health_chatbot.py --demo

Ask a custom question:
    python src/health_chatbot.py --question "How can I prevent dehydration?"

Environment variables:
    SPARK_BASE_URL  default http://127.0.0.1:30000/v1
    SPARK_API_KEY   default EMPTY
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
DEFAULT_API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health topics in simple, "
    "calm language that a person without medical training can follow. "
    "You do not diagnose, you do not prescribe, and you never give dosages or "
    "emergency instructions as if they were a medical decision. "
    "State clearly that your explanation is general education, list the warning "
    "signs that mean the person should seek urgent help, and recommend that the "
    "user contact a qualified clinician or local health service for any "
    "diagnosis, treatment or personal medical question. "
    "If the question describes an emergency, tell the user to contact local "
    "emergency services immediately. "
    "If information is missing, ask one short clarifying question instead of "
    "guessing."
)

DEMO_QUESTION = (
    "I came back from a trip three days ago and now I have a dry cough, a mild "
    "fever and I feel very tired. What should I know before I see a doctor?"
)

DISCLAIMER = (
    "Educational information only - not a diagnosis and not a treatment plan. "
    "Please consult a qualified clinician about your own health."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt always comes first and the new question always comes
    last, so the safety framing cannot be dropped by the caller.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("history entries must be dicts")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            raise ValueError("history entries need role 'user'/'assistant' and non-empty content")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def _post_json(url, payload, api_key, timeout=REQUEST_TIMEOUT):
    """POST a JSON payload and return the decoded JSON response."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", "Bearer %s" % api_key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach the Spark endpoint at %s (%s). "
            "Start a local OpenAI-compatible server for %s first." % (url, exc, MODEL_ID)
        )


def parse_reply(payload):
    """Extract the assistant text from an OpenAI-compatible response."""
    try:
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content")
    except (KeyError, IndexError, TypeError):
        raise ValueError("Unexpected response payload from the Spark endpoint")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Spark returned an empty answer")
    return content.strip()


def generate_reply(question, history=None, base_url=None, api_key=None, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS):
    """Send one question to Spark-X2.6-4B and return its plain-language answer."""
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": build_messages(question, history),
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "stream": False,
    }
    response = _post_json(url, payload, api_key or DEFAULT_API_KEY)
    return parse_reply(response)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plain-language health information chatbot on Spark-X2.6-4B")
    parser.add_argument("--question", help="question to ask the model")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo scenario")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible Spark endpoint")
    args = parser.parse_args(argv)

    question = args.question or DEMO_QUESTION
    try:
        answer = generate_reply(question, base_url=args.base_url)
    except (RuntimeError, ValueError) as exc:
        print("Error: %s" % exc)
        return 1

    print("Question: %s" % question)
    print()
    print(answer)
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
