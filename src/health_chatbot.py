"""Minimal patient-education chatbot demo built on Spark-X2.6-4B.

Talks to a locally hosted OpenAI-compatible endpoint (SGLang / vLLM / MLX /
llama.cpp server) serving ``XHToken/Spark-X2.6-4B``. Standard library only, so
the demo can run on a laptop without extra dependencies.

Usage:
    python src/health_chatbot.py "I have a cough and fever after travelling"

Environment:
    SPARK_BASE_URL  default http://127.0.0.1:30000/v1
    SPARK_API_KEY   default EMPTY (local servers usually ignore it)
    SPARK_MODEL     default XHToken/Spark-X2.6-4B (set to the served name)
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
DEFAULT_API_KEY = "EMPTY"

DISCLAIMER = (
    "This is general health information for education only, not a diagnosis "
    "or a treatment plan. Please contact a qualified clinician (doctor, nurse "
    "or pharmacist) about your own situation. In an emergency, contact your "
    "local emergency number immediately."
)

_REFERRAL_MARKERS = (
    "clinician",
    "doctor",
    "nurse",
    "pharmacist",
    "health worker",
    "healthcare professional",
    "health care professional",
    "emergency",
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who may have limited "
    "health literacy or limited access to care.\n"
    "Rules:\n"
    "1. Use short, plain sentences. Explain any medical word you must use.\n"
    "2. Do not diagnose, prescribe, or give drug doses. Do not claim certainty.\n"
    "3. Offer general, well-established information and list questions the "
    "person can ask a clinician.\n"
    "4. Point out warning signs that mean the person should seek care now, and "
    "tell them to contact local emergency services if those signs appear.\n"
    "5. If the question is outside health education, say so briefly.\n"
    "6. End by reminding the reader that this is education, not medical advice."
)

DEMO_QUESTION = (
    "I came back from a trip three days ago and now I have a dry cough, a "
    "fever of about 38.2 C, and I feel very tired. I do not have a doctor "
    "nearby. What should I know, and when should I get urgent help?"
)


def build_messages(question, history=None):
    """Return the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        if not content.strip():
            continue
        messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question, history=None, model=None, max_tokens=MAX_NEW_TOKENS,
                  temperature=0.2):
    """Build the OpenAI-compatible request body."""
    return {
        "model": model or os.environ.get("SPARK_MODEL", MODEL_ID),
        "messages": build_messages(question, history=history),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text from an OpenAI-compatible response."""
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response has no assistant content")
    return content.strip()


def ensure_safe(answer):
    """Append the education disclaimer when the model omitted a referral."""
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    text = answer.strip()
    lowered = text.lower()
    if any(marker in lowered for marker in _REFERRAL_MARKERS):
        return text
    return text + "\n\n" + DISCLAIMER


def chat(question, history=None, base_url=None, api_key=None, timeout=120,
         model=None, max_tokens=MAX_NEW_TOKENS):
    """Send one question to the local Spark-X2.6-4B endpoint and return text."""
    base = (base_url or os.environ.get("SPARK_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
    key = api_key or os.environ.get("SPARK_API_KEY", DEFAULT_API_KEY)
    payload = build_payload(
        question,
        history=history,
        model=model,
        max_tokens=max_tokens,
    )
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return ensure_safe(parse_response(data))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    try:
        answer = chat(question)
    except (urllib.error.URLError, OSError) as exc:
        print("Could not reach the Spark-X2.6-4B endpoint: %s" % exc, file=sys.stderr)
        print(
            "Start a local OpenAI-compatible server for %s and set "
            "SPARK_BASE_URL if it is not %s." % (MODEL_ID, DEFAULT_BASE_URL),
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        print("Invalid model response: %s" % exc, file=sys.stderr)
        return 1
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
