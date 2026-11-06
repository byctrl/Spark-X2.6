"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

This is a deliberately small, dependency-free demo for HER Hack-Astron #7.
It sends a plain-language health question to a locally served
``XHToken/Spark-X2.6-4B`` model through an OpenAI-compatible endpoint and
returns the model's patient-education answer.

Safety boundary: the assistant is an education aid only. It must not
diagnose, prescribe, or replace a clinician. The application never claims
that model output is medical ground truth; a human reads the answer.

Run:
    python src/health_chatbot.py "your question"

Environment:
    SPARK_BASE_URL  OpenAI-compatible base URL (default http://127.0.0.1:30000/v1)
    SPARK_API_KEY   API key if your server needs one (default "EMPTY")
    SPARK_MODEL     override model id (default XHToken/Spark-X2.6-4B)
    SPARK_TIMEOUT   request timeout in seconds (default 120)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_TIMEOUT = 120.0

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who may have limited "
    "health literacy or limited access to care. You run on Spark-X2.6-4B and "
    "only explain general health information in plain language.\n"
    "Rules you must follow:\n"
    "1. Never diagnose, never name a definitive condition, and never "
    "prescribe, dose, or stop a medicine.\n"
    "2. Explain general information, warning signs that need urgent care, and "
    "questions the person can ask a clinician.\n"
    "3. Use short sentences and plain words; avoid jargon or explain it.\n"
    "4. If the question suggests an emergency, tell the person to contact "
    "local emergency services or a clinician now.\n"
    "5. Say clearly that this is general education, not medical advice, and "
    "that a qualified clinician must decide on care.\n"
    "6. Do not ask for or repeat personal identifiers, and never guess a "
    "detail the person did not provide."
)

DEMO_QUESTION = (
    "I have had a cough and fever for three days, about a week after "
    "returning from a trip abroad. What should I do, and when should I get "
    "urgent help?"
)

DISCLAIMER = "General education only - not medical advice. See a qualified clinician."

def build_messages(
    question: str, history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Build the chat messages for one turn, keeping only valid history turns."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})
    return messages


def build_payload(
    messages: List[Dict[str, str]],
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Build a non-streaming OpenAI-compatible chat completion payload."""
    return {
        "model": model or MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }


def parse_completion(data: Any) -> str:
    """Extract the assistant text, failing loudly on unusable responses."""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected completion payload: no choices[0].message")

    content = message.get("content") if isinstance(message, dict) else None
    content = (content or "").strip()
    if not content:
        raise ValueError("model returned an empty answer")
    return content


def _post_json(url: str, payload: Dict[str, Any], api_key: str, timeout: float) -> Any:
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + str(api_key))
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError("Spark endpoint returned HTTP %s: %s" % (exc.code, detail)) from exc
    except URLError as exc:
        raise RuntimeError(
            "cannot reach Spark endpoint at %s: %s. Start a local "
            "OpenAI-compatible Spark server first." % (url, exc.reason)
        ) from exc
    return json.loads(raw)


def ask(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    transport: Optional[Callable[[str, Dict[str, Any], str, float], Any]] = None,
) -> str:
    """Send one question to Spark-X2.6-4B and return the plain-language answer."""
    base_url = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    if api_key is None:
        api_key = os.environ.get("SPARK_API_KEY", "EMPTY")
    if timeout is None:
        timeout = float(os.environ.get("SPARK_TIMEOUT", DEFAULT_TIMEOUT))
    if model is None:
        model = os.environ.get("SPARK_MODEL") or MODEL_ID

    payload = build_payload(build_messages(question, history), model=model)
    send = transport or _post_json
    data = send(base_url + "/chat/completions", payload, api_key, timeout)
    return parse_completion(data)


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    answer = ask(question)
    print("Q:", question)
    print()
    print("A:", answer)
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
