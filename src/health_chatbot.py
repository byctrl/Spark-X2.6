"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

The bot talks to a locally served Spark-X2.6-4B through an
OpenAI-compatible endpoint (llama.cpp / SGLang / vLLM / Spark-MLX-LLM).
It is an education assistant: it explains health topics in plain language
and never diagnoses, prescribes, or replaces a clinician.

Standard library only, so the demo runs without extra dependencies::

    python src/health_chatbot.py "What can I do about a mild cough?"

Endpoint configuration (environment variables):
    SPARK_BASE_URL   default http://127.0.0.1:30000/v1
    SPARK_MODEL      default XHToken/Spark-X2.6-4B
    SPARK_API_KEY    optional bearer token for the local server
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.2
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_TIMEOUT = 120.0

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information service. Explain health topics in simple, respectful, "
    "plain language for a general audience, including people with low "
    "health literacy or limited access to care.\n\n"
    "Hard rules:\n"
    "1. Do not diagnose. Do not name a condition as the user's condition.\n"
    "2. Do not prescribe medicines, give dosages, or tell the user to stop "
    "a prescribed treatment.\n"
    "3. Do not claim to replace a doctor, nurse, or pharmacist. Always "
    "encourage the user to consult a qualified clinician for personal "
    "decisions.\n"
    "4. If the description suggests an emergency, stop explaining and tell "
    "the user to contact local emergency services immediately.\n"
    "5. Say clearly when you are unsure, and ask for the information a "
    "clinician would need instead of guessing.\n"
    "6. Keep answers short: a plain summary, a few general self-care or "
    "preparation points, and when to seek care.\n"
)

SAFETY_NOTICE = (
    "This is general education, not medical advice. It cannot diagnose or "
    "treat any condition. Please consult a qualified health professional "
    "about your own situation."
)

URGENT_NOTICE = (
    "Possible emergency signs were detected in this message. Please contact "
    "your local emergency number or go to the nearest emergency department "
    "now. The information below is not a substitute for urgent care.\n"
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and still have a dry cough and "
    "some fever in the evening. What should I know before seeing a doctor?"
)

# Substring markers (English + Chinese) used only to attach an escalation
# notice. This is a coarse safety net, not a triage or diagnostic tool.
URGENT_PATTERNS = (
    "chest pain",
    "chest pressure",
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "shortness of breath",
    "severe bleeding",
    "uncontrolled bleeding",
    "passed out",
    "fainted",
    "unconscious",
    "suicidal",
    "kill myself",
    "end my life",
    "self-harm",
    "face drooping",
    "slurred speech",
    "sudden weakness",
    "severe head injury",
    "seizure",
    "胸痛",
    "呼吸困难",
    "大出血",
    "昏迷",
    "自杀",
    "自伤",
    "中风",
)

Transport = Callable[[str, Dict[str, Any], float], Dict[str, Any]]


@dataclass
class Reply:
    """A single chatbot answer plus the safety flags attached to it."""

    answer: str
    urgent: bool
    model: str


def is_urgent(text: str) -> bool:
    """Return True when the text contains an emergency marker."""

    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(pattern in lowered for pattern in URGENT_PATTERNS)


def build_messages(
    question: str, history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("history entries must be dicts")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant"):
            raise ValueError("history role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("history content must be a non-empty string")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def extract_content(response: Dict[str, Any]) -> str:
    """Pull the assistant text out of an OpenAI-compatible response."""

    if not isinstance(response, dict):
        raise ValueError("response must be a JSON object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = (message or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response has no assistant content")
    return content.strip()


def _post_json(
    url: str, payload: Dict[str, Any], timeout: float
) -> Dict[str, Any]:
    """POST JSON with the standard library and return the decoded reply."""

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    api_key = os.environ.get("SPARK_API_KEY")
    if api_key:
        request.add_header("Authorization", "Bearer " + api_key)

    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def chat(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    transport: Optional[Transport] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Reply:
    """Ask Spark-X2.6-4B one question and return a guarded reply."""

    messages = build_messages(question, history=history)
    resolved_model = model or os.environ.get("SPARK_MODEL") or MODEL_ID
    payload = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }

    endpoint = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL)
    url = endpoint.rstrip("/") + "/chat/completions"
    sender = transport or _post_json
    response = sender(url, payload, timeout)
    answer = extract_content(response)

    urgent = is_urgent(question)
    if urgent:
        answer = URGENT_NOTICE + "\n" + answer
    return Reply(answer=answer, urgent=urgent, model=resolved_model)


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION

    try:
        reply = chat(question)
    except (urllib.error.URLError, OSError) as exc:
        print("Could not reach the Spark endpoint: %s" % exc, file=sys.stderr)
        print(
            "Start a local OpenAI-compatible server and set SPARK_BASE_URL.",
            file=sys.stderr,
        )
        return 1
    except (ValueError, KeyError) as exc:
        print("Unexpected model response: %s" % exc, file=sys.stderr)
        return 1

    print("Question: %s" % question)
    print("Model: %s" % reply.model)
    print()
    print(reply.answer)
    print()
    print(SAFETY_NOTICE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
