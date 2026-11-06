"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Talks to a locally hosted OpenAI-compatible Spark-X2.6 endpoint
(SGLang / vLLM / llama.cpp server). Standard library only.

Safety boundary: the assistant gives plain-language *education* only.
It must not diagnose, prescribe, or replace a clinician; the system prompt
and the fixed footer below keep that boundary explicit.
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
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in plain, "
    "respectful language for a non-specialist reader.\n"
    "Rules:\n"
    "1. Never diagnose, never name a definitive condition, never prescribe or dose medicine.\n"
    "2. Give general education only: what the symptom usually means, what to observe, "
    "and which questions to ask a clinician.\n"
    "3. Always tell the reader to consult a qualified clinician about their own case.\n"
    "4. Say clearly when information may be incomplete or when urgent care is warranted.\n"
    "5. Do not ask for or repeat personal identifying details.\n"
    "Answer in the language of the question."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a cough and mild fever. "
    "What should I know before I see a doctor?"
)

URGENT_MARKERS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "shortness of breath",
    "uncontrolled bleeding",
    "fainting",
    "seizure",
    "suicidal",
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            raise ValueError("history turns need role user/assistant and non-empty content")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question, history=None, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS):
    """Build an OpenAI-compatible /chat/completions request body."""
    return {
        "model": model,
        "messages": build_messages(question, history),
        "max_tokens": int(max_tokens),
        "temperature": 0.3,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text, failing loudly on malformed output."""
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise ValueError("response has no choices[0]")

    message = choice.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned empty content")
    return content.strip()


def needs_urgent_care(text):
    lowered = (text or "").lower()
    return any(marker in lowered for marker in URGENT_MARKERS)


def add_safety_footer(text, question=""):
    """Append a fixed reminder; escalate instead when urgent wording appears."""
    if needs_urgent_care(text) or needs_urgent_care(question):
        note = (
            "Some of this may need urgent care. If symptoms are severe or getting worse, "
            "contact emergency services right away."
        )
    else:
        note = "This is general education, not a diagnosis. Please confirm with a qualified clinician."

    if note in text:
        return text
    return text.rstrip() + "\n\n" + note


def ask(question, base_url=DEFAULT_BASE_URL, model=MODEL_ID, history=None, timeout=REQUEST_TIMEOUT):
    """Send one question to the local Spark-X2.6-4B server and return the answer."""
    payload = build_payload(question, history=history, model=model)
    body = json.dumps(payload).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach Spark-X2.6 at {url}: {exc}") from exc

    return add_safety_footer(parse_response(data), question)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Health-information chatbot running on Spark-X2.6-4B (local endpoint)."
    )
    parser.add_argument("--question", default=None, help="question to ask (default: the demo question)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=MODEL_ID, help="served model name")
    parser.add_argument(
        "--show-payload",
        action="store_true",
        help="print the request payload and exit without any network call",
    )
    args = parser.parse_args(argv)

    question = args.question or DEMO_QUESTION

    if args.show_payload:
        print(json.dumps(build_payload(question, model=args.model), indent=2, ensure_ascii=False))
        return 0

    print(f"[model] {args.model} @ {args.base_url}")
    print(f"[you] {question}")
    print("[spark] " + ask(question, base_url=args.base_url, model=args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
