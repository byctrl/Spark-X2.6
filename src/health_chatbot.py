"""Minimal health-education chatbot on Spark-X2.6-4B (HER Hack-Astron #7).

Public-interest demo: inclusive, plain-language access to health information.
It only educates: it never diagnoses, prescribes or replaces a clinician, and
every reply points the reader to qualified professional care.

Quickstart with a local OpenAI-compatible server:

    python -m sglang.launch_server --model-path XHToken/Spark-X2.6-4B --port 30000
    python src/health_chatbot.py

Override the endpoint with SPARK_BASE_URL / SPARK_MODEL if your runtime differs.
"""

from __future__ import annotations

import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = """You are a patient-education assistant for people who may have low health literacy or limited access to care. Explain in plain language with short sentences, and use a numbered list when it helps.
Rules:
1. Do not diagnose, do not present a likely disease as a conclusion, and do not prescribe medicines or doses.
2. Say clearly what you do not know; never invent facts, statistics, guidelines or sources.
3. Give general, widely accepted information and list the questions the reader should ask a doctor, nurse or pharmacist.
4. End every reply with one sentence telling the reader to see a qualified clinician about their own situation.
5. Answer in the language of the question and keep it understandable to a non-specialist."""

DEMO_QUESTION = """I have had a cough and a mild fever for four days since I got back from a trip. What should I know before I see a doctor?"""

EMERGENCY_KEYWORDS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "severe bleeding",
    "unconscious",
    "suicide",
    "stroke",
)

EMERGENCY_NOTICE = """Important: if this might be an emergency, contact your local emergency number or the nearest emergency department now."""


def build_messages(question, history=None):
    """Return the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not content:
            raise ValueError("history turns need role=user|assistant and content")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def needs_emergency_notice(question):
    """Cheap keyword screen so urgent wording never waits for the model."""
    text = (question or "").lower()
    return any(word in text for word in EMERGENCY_KEYWORDS)


def _post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def generate_reply(question, history=None, base_url=None, model=None,
                   timeout=REQUEST_TIMEOUT, transport=None):
    """Ask Spark-X2.6-4B for a plain-language educational reply.

    ``transport`` is injectable for tests; it must accept
    ``(url, payload_dict, timeout)`` and return raw bytes or text.
    """
    messages = build_messages(question, history)
    base = base_url or os.environ.get("SPARK_BASE_URL", DEFAULT_BASE_URL)
    url = base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model or os.environ.get("SPARK_MODEL", MODEL_ID),
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.3,
        "stream": False,
    }

    raw = (transport or _post_json)(url, payload, timeout)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        body = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("endpoint did not return JSON") from exc

    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("model returned an empty answer")

    reply = content.strip()
    if needs_emergency_notice(question) and EMERGENCY_NOTICE not in reply:
        reply = EMERGENCY_NOTICE + "\n\n" + reply
    return reply


def main():
    print("Spark-X2.6-4B health-education demo")
    print("Model:", os.environ.get("SPARK_MODEL", MODEL_ID))
    print("Q:", DEMO_QUESTION)
    print("A:", generate_reply(DEMO_QUESTION))


if __name__ == "__main__":
    main()
