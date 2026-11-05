"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Educational information only: the assistant never diagnoses, prescribes,
or replaces a qualified health worker. Standard library only, so the demo
runs anywhere a local OpenAI-compatible Spark endpoint is reachable.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant inside a public-interest health "
    "information app. Answer in plain language, and answer in the language "
    "the user writes in. Explain what the described symptoms or terms "
    "generally mean, which questions to ask a clinician, and when to seek "
    "care. Never diagnose, never prescribe or adjust medication, never give "
    "dosages, and never claim to replace a qualified health worker. If the "
    "description suggests an emergency (chest pain, difficulty breathing, "
    "heavy bleeding, fainting, signs of stroke, thoughts of self-harm, or a "
    "pregnant person in pain or bleeding), tell the user to contact local "
    "emergency services immediately. Say plainly when you are unsure. Keep "
    "answers short (under 200 words) and end by reminding the user to consult "
    "a qualified health worker. This is educational information, not medical "
    "advice."
)

DEMO_QUESTION = (
    "Saya batuk dan demam sejak tiga hari lalu, setelah pulang dari "
    "perjalanan. Apa yang perlu saya perhatikan?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_payload(messages):
    """Build an OpenAI-compatible chat-completions payload."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
    }


def parse_response(data):
    """Extract assistant text, rejecting empty or malformed responses."""
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("model response contained no choices")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model response contained no text")
    return content.strip()


def _post_json(url, payload, timeout=TIMEOUT_SECONDS):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask(question, history=None, transport=None):
    """Send one question to the local Spark endpoint and return its answer."""
    payload = build_payload(build_messages(question, history))
    transport = transport or _post_json
    url = BASE_URL.rstrip("/") + "/chat/completions"
    return parse_response(transport(url, payload))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    print("Spark-X2.6-4B health-information demo (educational, not medical advice)")
    print("Model: %s" % MODEL_ID)
    print("Q: %s" % question)
    print()
    try:
        print(ask(question))
    except Exception as exc:  # surface setup problems instead of a traceback wall
        print("Could not reach the Spark endpoint at %s: %s" % (BASE_URL, exc))
        print("Start a local Spark-X2.6-4B server or set SPARK_BASE_URL.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
