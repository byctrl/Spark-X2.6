"""Minimal health-education chatbot demo for Spark-X2.6-4B.

This module contains no network code at import time. The demo can talk to any
OpenAI-compatible local server that hosts Spark-X2.6-4B.
"""

import json
import os
import urllib.error
import urllib.request


MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"

SYSTEM_PROMPT = (
    "You are a patient-education assistant. You help people understand health "
    "information in plain language. Always answer in the same language the user "
    "writes in. Explain any medical term you use.\n"
    "Hard rules:\n"
    "1. Do not diagnose, do not name a likely disease as the user's condition, "
    "and do not recommend prescription drugs, dosages, or treatments.\n"
    "2. Do not claim to replace a doctor, nurse, or pharmacist.\n"
    "3. Always tell the user which kind of clinician or service to contact and "
    "which warning signs mean they should seek urgent care now.\n"
    "4. If the user describes an emergency (for example chest pain, trouble "
    "breathing, heavy bleeding, fainting, or thoughts of self-harm), start the "
    "answer by telling them to contact local emergency services immediately.\n"
    "5. If information is missing, ask one short clarifying question instead of "
    "guessing.\n"
)

DEMO_QUESTION = (
    "I have had a cough and a fever for four days, and I flew back from another "
    "country last week. What should I do?"
)


def build_messages(question, history=None):
    """Build an OpenAI-style message list for one user question.

    ``history`` is an optional sequence of prior ``{"role": ..., "content": ...}``
    turns. The system prompt is always first and the new question is always last.
    """
    if not isinstance(question, str):
        raise TypeError("question must be a string")

    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for turn in history:
            if not isinstance(turn, dict):
                raise TypeError("history entries must be dicts")
            role = turn.get("role")
            content = turn.get("content")
            if role not in ("user", "assistant"):
                raise ValueError("history role must be 'user' or 'assistant'")
            if not isinstance(content, str):
                raise TypeError("history content must be a string")
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": cleaned_question})
    return messages


def _chat_completions_url(base_url):
    return base_url.rstrip("/") + "/chat/completions"


def generate_reply(question, base_url=None, model=MODEL_ID, timeout=120, history=None):
    """Send one question to a local OpenAI-compatible Spark-X2.6-4B server."""
    resolved_base_url = base_url or os.environ.get("SPARK_BASE_URL", DEFAULT_BASE_URL)
    payload = {
        "model": model,
        "messages": build_messages(question, history=history),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        _chat_completions_url(resolved_base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("Spark server returned HTTP %s: %s" % (exc.code, detail))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Spark server at %s: %s" % (resolved_base_url, exc.reason)
        )

    data = json.loads(body)
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected Spark response shape: %s" % body) from exc


def main():
    print("Spark-X2.6-4B health-education demo")
    print("Model:", MODEL_ID)
    print("Question:", DEMO_QUESTION)
    print()
    try:
        answer = generate_reply(DEMO_QUESTION)
    except RuntimeError as exc:
        print("Demo failed:", exc)
        print("Start a local OpenAI-compatible Spark-X2.6-4B server and retry.")
        return 1
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
