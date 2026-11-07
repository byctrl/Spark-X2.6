"""Minimal health-information chatbot powered by Spark-X2.6-4B.

Public-interest demo for HER Hack-Astron #7 (health information + accessibility).
The assistant explains health topics in plain language. It never diagnoses,
never prescribes, and always points the reader to a qualified clinician.
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
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who need clear, "
    "plain-language health information. Answer in the language of the question, "
    "using short sentences and everyday words. "
    "You must not diagnose, prescribe, interpret test results, or give dosages. "
    "If the question suggests an emergency (for example chest pain, trouble "
    "breathing, heavy bleeding, fainting, or a newborn with fever), say clearly "
    "that the person should seek emergency care now. "
    "Always remind the reader that this is general education, not a substitute "
    "for a qualified clinician who knows their situation."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak pulang dari perjalanan 5 hari lalu. "
    "Apa yang sebaiknya saya perhatikan sebelum ke klinik?"
)

DISCLAIMER = "This is general health education, not a diagnosis or a prescription."
REFERRAL = (
    "Please contact a qualified clinician or your local emergency service for "
    "advice about your own situation."
)


def build_messages(
    question: str, history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt always comes first and the current question always comes
    last, so the caller cannot accidentally replace the safety instructions.
    """
    text = (question or "").strip()
    if not text:
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = (turn or {}).get("role")
        content = ((turn or {}).get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})
    return messages


def ensure_safety_reply(reply: str) -> str:
    """Append the disclaimer and referral when the model left them out."""
    text = (reply or "").strip()
    lowered = text.lower()
    added = []
    if DISCLAIMER.lower() not in lowered:
        added.append(DISCLAIMER)
    if REFERRAL.lower() not in lowered:
        added.append(REFERRAL)
    if not text:
        return " ".join(added)
    if not added:
        return text
    return text + "\n\n" + " ".join(added)


def resolve_base_url(base_url: Optional[str] = None) -> str:
    url = base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL
    return url.rstrip("/")


def generate(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    base_url: Optional[str] = None,
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
    timeout: float = 120.0,
) -> str:
    """Send one question to a local OpenAI-compatible Spark-X2.6 server."""
    endpoint = resolve_base_url(base_url) + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": build_messages(question, history),
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            "could not reach Spark-X2.6 at {0}: {1}. Start a local "
            "OpenAI-compatible server (see FAQ.md) or set SPARK_BASE_URL.".format(endpoint, exc)
        ) from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("unexpected response from {0}: {1!r}".format(endpoint, data)) from exc
    return ensure_safety_reply(content)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plain-language health information chatbot using Spark-X2.6-4B."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEMO_QUESTION,
        help="Question to ask (defaults to the demo question).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:30000/v1",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the request messages and the reply as JSON.",
    )
    args = parser.parse_args(argv)

    messages = build_messages(args.question)
    try:
        reply = generate(args.question, base_url=args.base_url)
    except RuntimeError as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {"model": MODEL_ID, "messages": messages, "reply": reply},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
