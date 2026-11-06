"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Public-interest scenario: an inclusive, plain-language patient-education
assistant for people who have limited access to health information.
It does NOT diagnose, prescribe, adjust medication, or replace a clinician.

Spark-X2.6-4B is a text-only model (see FAQ.md), so no image input is used.
The model is expected to be served locally behind an OpenAI-compatible
endpoint, for example::

    python -m sglang.launch_server --model-path XHToken/Spark-X2.6-4B --port 30000

Run the demo::

    python src/health_chatbot.py
    python src/health_chatbot.py --question "..."

Environment:
    SPARK_BASE_URL  OpenAI-compatible base URL (default http://127.0.0.1:30000/v1)
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 120
MAX_HISTORY_TURNS = 6

DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public health information service.\n"
    "Rules:\n"
    "1. Use simple, everyday language (aim for a 6th-8th grade reading level) and short sentences.\n"
    "2. You do NOT diagnose, prescribe, adjust medication doses, or tell anyone to stop treatment.\n"
    "3. Explain general health concepts, help the reader understand what a symptom usually means, "
    "and list questions they can ask a clinician.\n"
    "4. Always end with a short, clear recommendation to consult a qualified clinician, and mention "
    "an emergency number when the symptoms could be urgent.\n"
    "5. Never ask for or repeat personal identifiers (name, ID number, address, phone number).\n"
    "6. If the request needs a diagnosis or a treatment decision, say plainly that only a clinician "
    "who examines the person can decide that.\n"
    "7. Do not claim to be a doctor and do not claim your answer is medical advice."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a cough and a mild fever in the "
    "evenings. What could this be, and what should I ask my doctor?"
)

# Very small red-flag list: these requests bypass the model entirely and get a
# fixed, human-written escalation message instead of generated text.
RED_FLAG_TERMS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "not breathing",
    "unconscious",
    "heavy bleeding",
    "suicidal",
    "kill myself",
    "seizure",
    "nyeri dada",
    "sesak napas berat",
    "胸痛",
    "呼吸困难",
)

RED_FLAG_MESSAGE = (
    "This may need urgent care. Please contact your local emergency number or go to the "
    "nearest emergency department now, or ask someone to take you there. "
    "This demo cannot assess emergencies."
)


def check_red_flags(question):
    """Return a fixed escalation message when the question looks urgent."""
    lowered = (question or "").lower()
    for term in RED_FLAG_TERMS:
        if term in lowered:
            return RED_FLAG_MESSAGE
    return None


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in list(history or [])[-MAX_HISTORY_TURNS:]:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})
    return messages


def build_payload(messages):
    """Build the OpenAI-compatible request body (non-streaming)."""
    if not messages:
        raise ValueError("messages must not be empty")
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text from an OpenAI-compatible response body."""
    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response has empty message content")
    return content.strip()


def chat(question, history=None, base_url=None, timeout=REQUEST_TIMEOUT):
    """Send one question to Spark-X2.6-4B and return the plain-language answer."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must be a non-empty string")

    red_flag = check_red_flags(text)
    if red_flag:
        return red_flag

    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    payload = build_payload(build_messages(text, history))
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError("Spark endpoint returned HTTP {}: {}".format(exc.code, detail)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach the Spark endpoint at {}. Start a local OpenAI-compatible "
            "server for {} first.".format(url, MODEL_ID)
        ) from exc

    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Spark endpoint returned a non-JSON response") from exc

    return parse_response(data)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot (educational demo, not medical advice)"
    )
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    args = parser.parse_args(argv)

    print("Model: {}".format(MODEL_ID))
    print("Endpoint: {}".format(args.base_url))
    print("Question: {}\n".format(args.question))
    print(chat(args.question, base_url=args.base_url))
    print("\n[Note] This demo is for health education only; it is not medical advice.")


if __name__ == "__main__":
    main()
