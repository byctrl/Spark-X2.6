#!/usr/bin/env python3
"""Health-information chatbot built on Spark-X2.6-4B (HER Hack-Astron #7).

Public-interest demo: an inclusive patient-education assistant that explains
health topics in plain language and points people to a clinician.

The script talks to any OpenAI-compatible server that is hosting
``XHToken/Spark-X2.6-4B`` (llama.cpp server, vLLM, SGLang, Ollama ``/v1`` ...):

    export SPARK_BASE_URL=http://127.0.0.1:30000/v1
    python health_chatbot.py --question "I have had a cough and fever for 3 days"

Safety boundaries enforced by this file, not only by the prompt:

* the model is asked for education only - no diagnosis, no prescriptions;
* red-flag symptoms get an urgent-care note prepended by application code;
* every answer ends with a fixed disclaimer and a referral to a clinician.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.2
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "It cannot diagnose or treat any condition - please see a qualified "
    "clinician about your own situation."
)

EMERGENCY_NOTE = (
    "If you have severe or worsening symptoms (for example chest pain, "
    "trouble breathing, heavy bleeding, fainting or a seizure), contact "
    "emergency services now instead of waiting for an online answer."
)

RED_FLAGS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "shortness of breath",
    "heavy bleeding",
    "unconscious",
    "seizure",
    "suicidal",
)

# Phrases that suggest the model crossed from education into diagnosis or
# prescription. They are reported to the caller as warnings; the fixed
# disclaimer is always added regardless.
RESTRICTED_PATTERNS = (
    "your diagnosis is",
    "i diagnose you",
    "you definitely have",
    "you should take",
    "you must take",
    "prescribe",
    "mg of",
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information chatbot. Reply in the language the user writes in, using "
    "short sentences and plain words a non-specialist can follow.\n"
    "Rules you must follow:\n"
    "1. Explain general health topics, common symptoms and when to seek care.\n"
    "2. Never diagnose, never name a likely disease as the user's condition, "
    "and never prescribe medicine, doses or treatments.\n"
    "3. Always tell the user to confirm anything important with a qualified "
    "clinician, and say clearly when a symptom needs urgent care.\n"
    "4. If information is missing, ask one short clarifying question instead "
    "of guessing.\n"
    "5. Do not claim to be a doctor and do not invent sources."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and a mild fever. What should I know before I see a doctor?"
)


def build_messages(question):
    """Return the chat messages sent to Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_payload(question):
    """Return the OpenAI-compatible request body for one question."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def extract_text(response):
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        text = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected response from Spark-X2.6 endpoint: %r" % (response,))
    text = (text or "").strip()
    if not text:
        raise ValueError("Spark-X2.6 returned an empty answer")
    return text


def check_answer(text):
    """Return the restricted phrases found in a model answer."""
    lowered = (text or "").lower()
    return [pattern for pattern in RESTRICTED_PATTERNS if pattern in lowered]


def compose_answer(text, question=""):
    """Add the application-level safety framing around a model answer."""
    warnings = check_answer(text)
    parts = []
    lowered = (question or "").lower()
    if any(flag in lowered for flag in RED_FLAGS):
        parts.append(EMERGENCY_NOTE)
    parts.append((text or "").strip())
    parts.append(DISCLAIMER)
    return "\n\n".join(parts), warnings


def post_json(url, payload, timeout=REQUEST_TIMEOUT):
    """Default transport: POST JSON to an OpenAI-compatible endpoint."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask(question, transport=None, base_url=None):
    """Ask Spark-X2.6-4B one health question and return the guarded answer."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    transport = transport or post_json
    base_url = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    payload = build_payload(question)
    response = transport(base_url + "/chat/completions", payload)
    raw_answer = extract_text(response)
    answer, warnings = compose_answer(raw_answer, question)
    return {
        "model": MODEL_ID,
        "question": question,
        "answer": answer,
        "raw_answer": raw_answer,
        "warnings": warnings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B plain-language health information chatbot"
    )
    parser.add_argument(
        "--question",
        default=DEMO_QUESTION,
        help="question to ask (default: the demo scenario)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:30000/v1",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full result as JSON"
    )
    args = parser.parse_args(argv)

    try:
        result = ask(args.question, base_url=args.base_url)
    except (urllib.error.URLError, OSError) as error:
        print("Could not reach the Spark-X2.6-4B endpoint: %s" % error, file=sys.stderr)
        print(
            "Start a local OpenAI-compatible server and set SPARK_BASE_URL, "
            "for example: export SPARK_BASE_URL=http://127.0.0.1:30000/v1",
            file=sys.stderr,
        )
        return 2
    except ValueError as error:
        print("Bad response from the endpoint: %s" % error, file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["answer"])
        for warning in result["warnings"]:
            print("[check] answer contains a restricted phrase: %s" % warning, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
