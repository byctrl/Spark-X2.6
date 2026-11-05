#!/usr/bin/env python3
"""Minimal health-information chatbot built on Spark-X2.6-4B.

HER Hack-Astron #7 / Spark-X2.6 Open Source AI for Good Challenge entry.

The chatbot answers general health-education questions in plain language and
always points the user back to a qualified clinician. It does not diagnose,
prescribe or replace professional care: the model output is an editable draft
for a person, never a decision.

The model is served by any OpenAI-compatible endpoint (vLLM, SGLang, Ollama,
Spark-MLX-LLM, ...). Point ``SPARK_BASE_URL`` at that endpoint; the default
matches the local port used in the repository quickstart.

Examples::

    python src/health_chatbot.py --demo
    python src/health_chatbot.py "My child has a fever, what should I watch for?"
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_TIMEOUT = 120
MAX_QUESTION_CHARS = 2000

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-health help desk. "
    "Answer in plain language at roughly a 6th-grade reading level. "
    "Explain what the described symptom or term generally means, what people "
    "commonly do next, and which warning signs mean 'seek care now'. "
    "Never diagnose, never name a specific prescription or dose, and never "
    "claim the answer is a medical decision. Always end with a short reminder "
    "to contact a qualified clinician or the local emergency number. "
    "If the question is unclear, ask one short clarifying question. "
    "If the user describes a possible emergency (chest pain, trouble "
    "breathing, heavy bleeding, thoughts of self-harm), stop and tell them to "
    "contact emergency services immediately."
)

DEMO_QUESTION = (
    "I came back from a trip four days ago and now I have a dry cough and a "
    "mild fever in the evening. I am 34 and otherwise healthy. What should I "
    "watch for, and when should I see a doctor?"
)

SAFETY_FOOTER = (
    "This is general health education, not a diagnosis. Please talk to a "
    "qualified clinician, or call your local emergency number if symptoms "
    "are severe."
)


def validate_question(question):
    """Normalise and bound a user question before it reaches the model."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    cleaned = " ".join(question.split())
    if not cleaned:
        raise ValueError("question must not be empty")
    return cleaned[:MAX_QUESTION_CHARS]


def build_messages(question):
    """Build the bounded chat messages sent to Spark-X2.6-4B."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": validate_question(question)},
    ]


def postprocess(reply):
    """Keep the model answer usable and never return an unsafe empty string."""
    text = (reply or "").strip()
    if not text:
        return "The model returned an empty answer. " + SAFETY_FOOTER
    return text


def extract_reply(body):
    """Read ``choices[0].message.content`` from an OpenAI-compatible response."""
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response from model endpoint: %r" % (body,)) from exc


def generate(question, base_url=None, model=MODEL_ID, timeout=DEFAULT_TIMEOUT):
    """Send one question to the local Spark-X2.6-4B server and return the text."""
    base_url = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    payload = {
        "model": model,
        "messages": build_messages(question),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
        "stream": False,
    }
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return postprocess(extract_reply(body))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a plain-language health-education question."
    )
    parser.add_argument("question", nargs="?", default=None, help="question to ask")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo question")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint URL")
    args = parser.parse_args(argv)

    question = DEMO_QUESTION if (args.demo or not args.question) else args.question
    try:
        answer = generate(question, base_url=args.base_url)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print("Could not reach the Spark-X2.6-4B endpoint: %s" % exc, file=sys.stderr)
        print("Start a local OpenAI-compatible server and set SPARK_BASE_URL.", file=sys.stderr)
        return 1

    print("Q:", validate_question(question))
    print()
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
