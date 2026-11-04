"""Minimal health-information chatbot powered by Spark-X2.6-4B.

Purpose: an inclusive, plain-language gateway to health knowledge for people
with limited access to a clinician. The chatbot is an educational demo only.
It does not diagnose, prescribe, or replace professional care, and the output
is shown to the user with an explicit disclaimer.

Runtime: any OpenAI-compatible server exposing Spark-X2.6-4B, for example
SGLang/vLLM on http://127.0.0.1:30000/v1 (override with --base-url or the
SPARK_BASE_URL environment variable).

Usage:
    python src/health_chatbot.py
    python src/health_chatbot.py "Is a mild fever after travel serious?"
"""

import argparse
import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

DISCLAIMER = (
    "Educational information only. This is not a diagnosis or a prescription. "
    "Please consult a qualified health professional about your situation."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who may have limited "
    "access to health information. Explain things in plain language at about a "
    "6th-grade reading level, in short paragraphs. You must not diagnose, "
    "prescribe, or suggest a specific dose of any medicine. Say clearly when "
    "in-person care from a clinician is needed, and always encourage the user "
    "to consult a qualified health professional. If the question suggests an "
    "emergency (for example chest pain, trouble breathing, heavy bleeding, "
    "fainting, or thoughts of self-harm), first tell the user to seek emergency "
    "care immediately. Do not ask for or repeat personal identifiers."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and now I have a cough and a mild "
    "fever. What could it be and what should I do?"
)


def build_messages(question, system_prompt=SYSTEM_PROMPT):
    """Return the chat messages for one user question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]


def build_payload(question, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body."""
    return {
        "model": model,
        "messages": build_messages(question),
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }


def _http_post_json(url, payload, timeout=REQUEST_TIMEOUT):
    """POST JSON with the standard library and return the decoded response."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_text(response):
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "model response is missing choices[0].message.content"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model response content is empty")
    return content.strip()


def chat(question, base_url=None, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS,
         transport=None):
    """Ask Spark-X2.6-4B one question and return the plain-language answer."""
    resolved_base = (
        base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL
    ).rstrip("/")
    send = transport or _http_post_json
    payload = build_payload(question, model=model, max_tokens=max_tokens)
    response = send(resolved_base + "/chat/completions", payload, REQUEST_TIMEOUT)
    return extract_text(response)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B for plain-language health information."
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible base URL of the Spark server")
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args(argv)

    print(DISCLAIMER)
    print()
    print(chat(args.question, base_url=args.base_url, model=args.model))


if __name__ == "__main__":
    main()
