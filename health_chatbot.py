"""Minimal AI-for-Good demo: inclusive health-information chatbot on Spark-X2.6-4B.

The application is deliberately small and dependency-free (Python standard
library only). It sends a plain-language question to a *locally hosted*
OpenAI-compatible endpoint serving ``XHToken/Spark-X2.6-4B`` (SGLang, vLLM or
llama.cpp server) and returns an educational answer.

What the model does in this application:
  * explains the described signs in simple, inclusive language
  * never diagnoses, prescribes or replaces a clinician
  * always points the reader at a qualified professional or local service

Run against a local endpoint::

    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python health_chatbot.py "I have a cough"

Without a running endpoint the demo still prints the exact prompt that would
be sent, so the wiring stays inspectable offline. Model output is never
treated as ground truth: the reader (or a trained coordinator) is the gate.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")
MAX_NEW_TOKENS = 1024
MAX_QUESTION_CHARS = 2000
TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = """You are a patient-education assistant for a public-interest
health information service. Your readers include people with low health
literacy, limited internet access, disabilities and non-native speakers.

Rules:
1. Use short sentences and plain language; expand every medical term you use.
2. Do not diagnose, do not present a likely disease as a conclusion, and do
   not recommend prescription drugs, dosages or home remedies.
3. Explain in general terms what the described sign may mean, what the person
   can safely do next, and which red-flag signs mean "seek care now".
4. Always tell the reader to confirm with a qualified clinician or local
   health service, and state that this is education, not medical advice.
5. If the question describes an emergency or mentions self-harm, lead with
   the advice to contact local emergency services immediately.
6. Never ask for or repeat personal identifiers."""

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and now I have a dry cough and a "
    "fever in the evening. I do not have insurance and I am not sure whether "
    "I should wait. What should I do?"
)

DISCLAIMER = (
    "\n\n---\n"
    "This is general health education, not medical advice or a diagnosis. "
    "Please confirm with a qualified clinician or your local health service."
)

DISCLAIMER_MARKER = "not medical advice"


class ChatbotError(RuntimeError):
    """Raised when the local Spark endpoint cannot produce a usable answer."""


def validate_question(question: str) -> str:
    """Normalise and bound a user question so the prompt stays predictable."""
    if not isinstance(question, str):
        raise ChatbotError("question must be a string")
    text = " ".join(question.split())
    if not text:
        raise ChatbotError("question must not be empty")
    if len(text) > MAX_QUESTION_CHARS:
        raise ChatbotError(
            f"question is too long ({len(text)} > {MAX_QUESTION_CHARS} characters)"
        )
    return text


def build_messages(question: str, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": " ".join(content.split())})
    messages.append({"role": "user", "content": validate_question(question)})
    return messages


def build_payload(question: str, history=None) -> dict:
    """Build the OpenAI-compatible request body for the local Spark server."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
        "stream": False,
    }


def extract_text(response) -> str:
    """Pull the final answer out of an OpenAI-style response, or fail loudly.

    A schema-valid response with an empty ``content`` means the thinking or
    tool-call budget was exhausted before an answer was delivered; that is an
    error here, not a successful turn.
    """
    if not isinstance(response, dict):
        raise ChatbotError("endpoint returned a non-object response")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ChatbotError("endpoint returned no choices")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        finish = choice.get("finish_reason") or "unknown"
        raise ChatbotError(
            f"endpoint returned no final answer (finish_reason={finish})"
        )
    return content.strip()


def with_disclaimer(answer: str) -> str:
    """Guarantee the safety disclaimer is present exactly once."""
    text = answer.strip()
    if DISCLAIMER_MARKER in text.lower():
        return text
    return text + DISCLAIMER


def _http_transport(payload: dict) -> dict:
    """POST the payload to the local OpenAI-compatible endpoint."""
    url = BASE_URL.rstrip("/") + "/chat/completions"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise ChatbotError(f"endpoint HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ChatbotError(f"cannot reach {url}: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ChatbotError("endpoint returned invalid JSON") from exc


def chat(question: str, history=None, transport=None) -> str:
    """Answer one health question; ``transport`` is injectable for tests."""
    payload = build_payload(question, history)
    send = transport or _http_transport
    return with_disclaimer(extract_text(send(payload)))


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args) if args else DEMO_QUESTION
    print(f"model:    {MODEL_ID}")
    print(f"endpoint: {BASE_URL}\n")
    print(f"question: {question}\n")
    try:
        answer = chat(question)
    except ChatbotError as exc:
        print(f"[offline demo] no answer from the local endpoint: {exc}\n")
        print("prompt that would be sent:\n")
        print(json.dumps(build_payload(question), indent=2, ensure_ascii=False))
        return 1
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
