"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

Scope: inclusive, plain-language health *education*. The model explains and
signposts; it does not diagnose, prescribe, or replace a clinician. This is a
small, runnable demo of Spark-X2.6-4B in a public-interest setting.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public health information "
    "service. You support people with low health literacy and people using "
    "assistive technology.\n"
    "Rules:\n"
    "1. Use short sentences and plain language; expand any medical term you "
    "must use.\n"
    "2. Do not diagnose, do not name a likely disease as the user's condition, "
    "and do not prescribe or adjust medicines or doses.\n"
    "3. Explain general options, what to observe, and when to seek care.\n"
    "4. Always tell the user to consult a qualified clinician for decisions "
    "about their own health.\n"
    "5. If the description suggests an emergency, say clearly that urgent "
    "in-person care is needed now.\n"
    "6. Never ask for or repeat personal identifiers; if the user shares them, "
    "tell them not to and continue without them.\n"
    "7. If you are unsure, say so instead of guessing."
)

DEMO_QUESTION = (
    "I got back from a trip three days ago and since then I have a dry cough "
    "and a mild fever. What should I pay attention to?"
)

EMERGENCY_SIGNS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "difficulty breathing",
    "severe bleeding",
    "unconscious",
    "seizure",
    "suicidal",
    "stiff neck",
)

URGENT_NOTICE = (
    "This sounds like it may need urgent in-person care. Contact your local "
    "emergency number or go to the nearest emergency department now. This "
    "chat cannot assess an emergency."
)

DISCLAIMER = (
    "This is general health information, not medical advice. Please talk to a "
    "qualified clinician about your own situation."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def needs_urgent_care(text):
    """Cheap keyword screen, run before the model so emergencies are not delayed."""
    lowered = (text or "").lower()
    return any(sign in lowered for sign in EMERGENCY_SIGNS)


def format_reply(question, answer):
    """Attach the urgent-care banner when needed and always add the disclaimer."""
    parts = []
    if needs_urgent_care(question):
        parts.append(URGENT_NOTICE)
    answer = (answer or "").strip()
    if answer:
        parts.append(answer)
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def query_spark(messages, base_url=None, model=None, max_tokens=MAX_NEW_TOKENS, timeout=120):
    """Call a local OpenAI-compatible Spark endpoint and return the reply text."""
    root = base_url or os.environ.get("SPARK_BASE_URL", DEFAULT_BASE_URL)
    url = root.rstrip("/") + "/chat/completions"
    payload = {
        "model": model or MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError("Spark request failed (%s): %s" % (url, exc)) from exc
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected Spark response: %r" % (body,)) from exc


def main(argv=None):
    """Run one demo question against the local Spark endpoint."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    answer = query_spark(build_messages(question))
    print(format_reply(question, answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
