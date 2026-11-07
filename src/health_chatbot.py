"""Minimal patient-education chatbot powered by Spark-X2.6-4B.

This is the smallest working application for the HER Hack-Astron #7
"Open Source AI for Good" challenge: an inclusive health-knowledge
assistant that runs entirely against a locally hosted Spark-X2.6-4B
OpenAI-compatible endpoint.

What the model does here: it reads a person's health question and writes
a plain-language educational answer (what is generally known, what to
watch for, which questions to ask a clinician). It never diagnoses or
prescribes, and it never reaches the network beyond the configured
local endpoint.

Run the demo:
    python src/health_chatbot.py
    python src/health_chatbot.py --base-url http://127.0.0.1:30000/v1 \
        --question "What should I ask my doctor about my asthma?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
REQUEST_TIMEOUT = 120.0

SYSTEM_PROMPT = (
    "You are a patient-education assistant running on the open-source "
    "Spark-X2.6-4B language model. Help people understand general health "
    "information in plain, respectful language.\n"
    "Rules:\n"
    "1. Do not diagnose, do not prescribe, and do not give dosages; you do not "
    "replace a clinician.\n"
    "2. If the question suggests an emergency, tell the person to contact "
    "local emergency services right away.\n"
    "3. Explain what is generally known, why it matters, and which questions "
    "the person should ask a clinician.\n"
    "4. State clearly when you are unsure, and never invent sources.\n"
    "5. Answer in the language the person used."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and still have a dry cough and a "
    "mild fever. What should I pay attention to before I see a doctor?"
)

SAFETY_NOTE = (
    "This is general health education, not a diagnosis or a prescription. "
    "Please contact a qualified clinician or your local emergency service for "
    "your own situation."
)

_REFERRAL_MARKERS = (
    "clinician",
    "doctor",
    "physician",
    "pharmacist",
    "nurse",
    "emergency",
    "professional",
    "klinisi",
    "dokter",
    "医生",
    "医师",
    "急诊",
)


def build_messages(
    question: str, history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(
    messages: List[Dict[str, str]],
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Build an OpenAI-compatible chat-completions payload."""
    return {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }


def extract_reply(data: Dict[str, Any]) -> str:
    """Pull the assistant text out of a chat-completions response."""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response payload from the Spark endpoint") from exc

    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("the model returned an empty reply")
    return content.strip()


def with_safety_note(answer: str) -> str:
    """Append the education-only disclaimer when no referral is present."""
    text = (answer or "").strip()
    if not text:
        return SAFETY_NOTE

    lowered = text.lower()
    if any(marker in lowered for marker in _REFERRAL_MARKERS):
        return text
    return text + "\n\n" + SAFETY_NOTE


def _post_json(
    url: str, payload: Dict[str, Any], timeout: float = REQUEST_TIMEOUT
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def chat(
    question: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = MODEL_ID,
    history: Optional[List[Dict[str, str]]] = None,
    timeout: float = REQUEST_TIMEOUT,
) -> str:
    """Ask Spark-X2.6-4B a health question and return the educational answer."""
    messages = build_messages(question, history)
    payload = build_payload(messages, model=model)
    url = base_url.rstrip("/") + "/chat/completions"
    data = _post_json(url, payload, timeout=timeout)
    return with_safety_note(extract_reply(data))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Minimal Spark-X2.6-4B patient-education chatbot"
    )
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL"
    )
    parser.add_argument("--model", default=MODEL_ID, help="served model name")
    args = parser.parse_args(argv)

    try:
        answer = chat(args.question, base_url=args.base_url, model=args.model)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"error: no reply from {args.base_url}: {exc}", file=sys.stderr)
        print(
            "hint: serve Spark-X2.6-4B locally, for example "
            "`python -m sglang.launch_server --model-path XHToken/Spark-X2.6-4B`",
            file=sys.stderr,
        )
        return 2

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
