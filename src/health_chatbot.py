"""Minimal, safety-bounded health-information chatbot for Spark-X2.6-4B.

Public-interest demo for HER Hack-Astron #7: plain-language health education
for people who cannot easily reach a clinician. Spark-X2.6-4B is served
locally behind an OpenAI-compatible endpoint (vLLM, SGLang, llama.cpp server),
so this module has no third-party dependencies and sends nothing to a cloud
service.

The application is intentionally bounded: it explains general health topics,
never diagnoses or prescribes, and intercepts emergency language before any
model call.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Callable, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
MODEL_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in "
    "simple, respectful language for a reader with no medical training.\n"
    "Rules:\n"
    "1. Never diagnose, never prescribe, and never give a dose or a treatment plan.\n"
    "2. Say clearly that this is general education, not medical advice.\n"
    "3. Encourage the reader to see a qualified clinician, and name the type "
    "of clinician or service when that is useful.\n"
    "4. Do not ask for or repeat names, addresses, ID numbers, or other "
    "personal data.\n"
    "5. If the message suggests an emergency, tell the reader to contact "
    "emergency services now.\n"
    "6. If you are unsure, say so instead of guessing.\n"
)

DEMO_QUESTION = (
    "I have had a cough and mild fever for three days since returning from a "
    "trip. What should I know before I see a doctor?"
)

DISCLAIMER = (
    "General health education only - not a diagnosis or treatment plan. "
    "Please consult a qualified clinician about your own situation."
)

FALLBACK = (
    "The model returned an empty answer. Please rephrase your question or try "
    "again shortly."
)

EMERGENCY_PATTERNS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "severe bleeding",
    "suicid",
    "overdose",
    "\u80f8\u75db",
    "\u547c\u5438\u56f0\u96be",
    "\u81ea\u6740",
)

EMERGENCY_NOTICE = (
    "This sounds like a possible emergency. Please contact your local "
    "emergency number or go to the nearest emergency department now. "
    "This demo cannot provide urgent care."
)


def build_messages(
    question: str, history: Optional[List[dict]] = None
) -> List[dict]:
    """Build the chat payload, keeping only valid prior turns."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": (question or "").strip()})
    return messages


def emergency_notice(text: str) -> Optional[str]:
    """Return an urgent-care message for red-flag wording, else None."""
    lowered = (text or "").lower()
    for pattern in EMERGENCY_PATTERNS:
        if pattern in lowered:
            return EMERGENCY_NOTICE
    return None


def _post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(
            "Could not reach Spark-X2.6-4B at "
            f"{MODEL_BASE_URL}. Start a local OpenAI-compatible server and "
            f"set SPARK_BASE_URL. Original error: {exc}"
        ) from exc


def default_transport(messages: List[dict]) -> str:
    """Call the local Spark-X2.6-4B OpenAI-compatible chat endpoint."""
    url = MODEL_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
    }
    body = _post_json(url, payload)
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected response from model server: {body!r}"
        ) from exc


def answer(
    question: str,
    history: Optional[List[dict]] = None,
    transport: Optional[Callable[[List[dict]], str]] = None,
) -> str:
    """Answer one health-education question with safety checks applied."""
    text = (question or "").strip()
    if not text:
        return "Please type a question about a health topic you would like explained."

    notice = emergency_notice(text)
    if notice:
        return notice

    call = transport or default_transport
    reply = (call(build_messages(text, history)) or "").strip()
    if not reply:
        return FALLBACK + "\n\n" + DISCLAIMER
    return reply + "\n\n" + DISCLAIMER


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print("usage: python health_chatbot.py [question]")
        print(f"model: {MODEL_ID} via {MODEL_BASE_URL}")
        return 0

    question = " ".join(args) if args else DEMO_QUESTION
    print(f"Q: {question}\n")
    try:
        print(answer(question))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
