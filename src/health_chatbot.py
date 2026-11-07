"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

The chatbot is a patient-education assistant: it explains health topics in
plain language and always points the user to a qualified clinician. It never
diagnoses and never prescribes. It talks to any OpenAI-compatible server that
is serving XHToken/Spark-X2.6-4B.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
DEFAULT_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public health information service.\n"
    "Rules:\n"
    "1. Explain health topics in simple, plain language and short paragraphs.\n"
    "2. Never give a diagnosis, a prescription, a dosage, or a treatment plan.\n"
    "3. Say clearly when something needs a qualified clinician and how soon.\n"
    "4. If the situation may be an emergency, tell the user to contact local "
    "emergency services now.\n"
    "5. Do not invent facts, sources, or statistics. Say clearly what you do not know.\n"
    "6. Keep the answer short enough to read on a phone."
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "For diagnosis or treatment, contact a qualified health worker."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough and "
    "a mild fever in the evening. I do not have a family doctor. What should I do?"
)

EMERGENCY_KEYWORDS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "difficulty breathing",
    "heavy bleeding",
    "unconscious",
    "seizure",
    "suicide",
    "overdose",
)

EMERGENCY_REPLY = (
    "This may be an emergency. Please contact your local emergency number now. "
    "If you cannot reach it, go to the nearest emergency department. "
    "I cannot assess emergency symptoms."
)


def is_emergency(text: str) -> bool:
    """Return True when the message may describe an emergency situation."""
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


def build_messages(question: str, history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": (question or "").strip()})
    return messages


def build_payload(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, object]:
    """Build the OpenAI-compatible request body for Spark-X2.6-4B."""
    return {
        "model": model,
        "messages": build_messages(question, history),
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False,
    }


def parse_answer(response: Dict[str, object]) -> str:
    """Extract the assistant text from an OpenAI-compatible response."""
    choices = (response or {}).get("choices") or []
    if not choices:
        raise ValueError("model response has no choices")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise ValueError("model response has empty content")
    return content


def chat(
    question: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = MODEL_ID,
    timeout: int = DEFAULT_TIMEOUT,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Answer a health question with a locally served Spark-X2.6-4B model."""
    if is_emergency(question):
        return EMERGENCY_REPLY
    payload = build_payload(question, history=history, model=model)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(
            "cannot reach the Spark-X2.6-4B endpoint at %s: %s" % (base_url, exc)
        ) from exc
    return parse_answer(body)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot demo"
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the request payload and exit (no model call)",
    )
    args = parser.parse_args(argv)

    if args.show_prompt:
        print(json.dumps(build_payload(args.question), indent=2, ensure_ascii=False))
        return 0

    print("Question: %s\n" % args.question)
    try:
        answer = chat(args.question, base_url=args.base_url, model=args.model)
    except (ConnectionError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1

    print("Spark-X2.6-4B: %s\n" % answer)
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
