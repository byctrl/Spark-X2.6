"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Sends one plain-language health question to a locally hosted,
OpenAI-compatible Spark-X2.6-4B endpoint and prints an educational answer.
It never diagnoses and never prescribes: every answer is education only and
ends by pointing the user to a clinician.

Usage:
    python src/health_chatbot.py                  # run DEMO_QUESTION
    python src/health_chatbot.py "my question"    # ask your own question

Environment:
    SPARK_BASE_URL   default http://127.0.0.1:30000/v1
    SPARK_API_KEY    default "EMPTY" (local servers usually ignore it)
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

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-health information line. "
    "Explain health topics in simple, plain language at a 6th-grade reading level.\n"
    "Rules you must always follow:\n"
    "1. Do NOT diagnose. Never say what condition the person has.\n"
    "2. Do NOT prescribe or recommend doses, antibiotics, or any specific medicine.\n"
    "3. Do NOT replace a clinician. End with a short, concrete next step "
    "(who to contact, and which warning signs mean seek care now).\n"
    "4. If the question mentions emergency warning signs (chest pain, trouble "
    "breathing, fainting, heavy bleeding, confusion, suicidal thoughts), say "
    "clearly and first that they should contact emergency services now.\n"
    "5. Say when you are unsure, and never invent sources, statistics or local "
    "phone numbers.\n"
    "6. Keep the answer under 250 words and use short sentences."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and some fever in the evenings. What should I know?"
)

DISCLAIMER = "This is general education, not medical advice. See a clinician."

EMERGENCY_TERMS = (
    "chest pain",
    "trouble breathing",
    "can't breathe",
    "cannot breathe",
    "heavy bleeding",
    "fainting",
    "suicidal",
)


def needs_emergency_notice(question: str) -> bool:
    """True when the question mentions an emergency warning sign."""
    lowered = (question or "").lower()
    return any(term in lowered for term in EMERGENCY_TERMS)


def build_messages(question: str) -> list:
    """Build the chat messages sent to Spark-X2.6-4B for one question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_payload(question: str) -> dict:
    """Build the OpenAI-compatible request body for Spark-X2.6-4B."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
    }


def extract_answer(response: dict) -> str:
    """Pull the assistant text out of a chat-completions response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected Spark response shape: {response!r}") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Spark returned an empty answer")
    return content.strip()


def _default_transport(url: str, payload: dict, api_key: str) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", "Bearer " + api_key)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(
            f"Spark endpoint returned HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Spark endpoint at {url}. Start a local "
            f"OpenAI-compatible server first (see README quickstart) and set "
            f"SPARK_BASE_URL. Cause: {exc.reason}"
        ) from exc


def ask(question: str, base_url: str = None, api_key: str = None, transport=None) -> str:
    """Ask Spark-X2.6-4B one question and return the answer text."""
    base_url = (
        base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL
    ).rstrip("/")
    api_key = api_key or os.environ.get("SPARK_API_KEY") or "EMPTY"
    transport = transport or _default_transport
    payload = build_payload(question)
    response = transport(base_url + "/chat/completions", payload, api_key)
    return extract_answer(response)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print(f"model : {MODEL_ID}")
    print(f"asking: {question}\n")
    if needs_emergency_notice(question):
        print("NOTE  : This mentions a possible emergency. Contact local ")
        print("        emergency services now rather than waiting for this tool.\n")
    try:
        answer = ask(question)
    except (RuntimeError, ValueError) as exc:
        print(f"error : {exc}", file=sys.stderr)
        return 1
    print(answer)
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
