"""Health-information chatbot demo built on Spark-X2.6-4B.

The application is deliberately small: it turns a plain-language health
question into an OpenAI-compatible chat request for a locally hosted
Spark-X2.6-4B endpoint and prints the reply.

What the model does here: it explains general health information in simple
language, without diagnosing, naming a condition for the user, or
recommending medicines. Every reply has to point the user to a qualified
clinician or emergency service.

Run:
    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python health_chatbot.py --demo
    python health_chatbot.py --question "What is a fever?"
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
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information service. Answer in the language of the user's question, "
    "using short sentences and plain words. You explain general health "
    "information only: you never diagnose, you never name a disease as the "
    "user's own condition, you never recommend or dose medicines, and you "
    "never tell someone to delay care. Always state clearly when the user "
    "should contact a qualified clinician or an emergency service. If the "
    "question needs personal medical details you do not have, ask for those "
    "details to be checked by a clinician instead of guessing. Finish with "
    "one line starting with 'Next step:' that names the safest concrete "
    "action available to the user."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and fever in the evening. I have no insurance and I am not sure whether "
    "this is serious. What should I know before I see someone?"
)

DISCLAIMER = (
    "This demo gives general health information only and is not medical "
    "advice. For urgent symptoms contact local emergency services."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError(
                "history entries need role user/assistant and string content"
            )
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question, history=None, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body for the local Spark server."""
    return {
        "model": model,
        "messages": build_messages(question, history),
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }


def extract_reply(data):
    """Read the assistant text from an OpenAI-compatible response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response shape from Spark endpoint") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Spark endpoint returned an empty reply")
    return content.strip()


def ask(question, base_url=DEFAULT_BASE_URL, model=MODEL_ID, history=None, opener=None):
    """Send one question to Spark-X2.6-4B and return its reply text."""
    payload = build_payload(question, history=history, model=model)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(request, timeout=REQUEST_TIMEOUT) as response:
            raw = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "could not reach the Spark endpoint at %s (%s); start a local "
            "OpenAI-compatible server for %s first" % (base_url, exc, MODEL_ID)
        ) from exc
    return extract_reply(json.loads(raw.decode("utf-8")))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health chatbot demo")
    parser.add_argument("--question", help="health question to ask")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo question")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SPARK_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible Spark base URL",
    )
    parser.add_argument("--model", default=MODEL_ID, help="served model name")
    args = parser.parse_args(argv)

    question = args.question or (DEMO_QUESTION if args.demo else None)
    if not question:
        parser.error("give --question or --demo")

    print(DISCLAIMER)
    print()
    print(ask(question, base_url=args.base_url, model=args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
