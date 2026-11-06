"""Minimal Spark-X2.6-4B chatbot for inclusive health knowledge.

Public-interest demo for HER Hack-Astron #7 (AI for Good): a plain-language
patient-education assistant built on XHToken/Spark-X2.6-4B.

Safety boundaries (enforced in the system prompt and in code):
  * education only - no diagnosis, no prescriptions, no dosages;
  * always point the user to a qualified clinician;
  * emergency wording gets a fixed, code-generated notice prepended to the
    model answer, so it does not depend on the model behaving correctly.

Usage:
    python src/health_chatbot.py "my question"
    python src/health_chatbot.py            # runs DEMO_QUESTION

The model is served locally through an OpenAI-compatible endpoint
(SGLang / vLLM / llama.cpp / MLX). Override SPARK_BASE_URL to point at it.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")
TIMEOUT = float(os.environ.get("SPARK_TIMEOUT", "120"))

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people with limited health "
    "literacy. Explain in plain language (about a grade 6-8 reading level) "
    "using short sentences and everyday words. "
    "You must NOT diagnose, must NOT prescribe, and must NOT give dosages or "
    "treatment plans. Say clearly that this is general education, not medical "
    "advice, and that only a qualified clinician can assess their situation. "
    "Always tell the user when and how to contact a clinician or a local "
    "emergency service. If the message suggests an emergency, tell them to seek "
    "emergency care right now. If important details are missing, ask one short "
    "clarifying question instead of guessing. Never ask for names, ID numbers, "
    "or other personal data."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a cough and "
    "fever in the evening. I do not have much money and I am not sure if I "
    "should see a doctor. What could this be?"
)

EMERGENCY_MARKERS = (
    "chest pain",
    "chest pressure",
    "cannot breathe",
    "can't breathe",
    "trouble breathing",
    "shortness of breath",
    "heavy bleeding",
    "severe bleeding",
    "unconscious",
    "fainted",
    "fainting",
    "seizure",
    "stroke",
    "suicide",
    "suicidal",
    "kill myself",
    "harm myself",
)

REFERRAL_MARKERS = (
    "doctor",
    "clinician",
    "physician",
    "nurse",
    "pharmacist",
    "health worker",
    "clinic",
    "hospital",
    "emergency",
    "professional",
)

EMERGENCY_NOTICE = (
    "Safety notice: your message may describe a medical emergency. Please "
    "contact your local emergency number or go to the nearest emergency "
    "department now. This chatbot cannot help with emergencies."
)


def build_messages(question, history=None):
    """Build the chat message list for Spark-X2.6-4B.

    ``history`` is an optional list of previous ``{"role", "content"}`` turns;
    it is copied, never mutated.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": str(question).strip()})
    return messages


def build_payload(question, history=None, max_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible chat-completions request body."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.9,
        "stream": False,
    }


def extract_text(response):
    """Pull the assistant text out of a chat-completion response body."""
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return (content or "").strip()


def looks_like_emergency(text):
    """Cheap keyword screen; runs before the model is ever called."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in EMERGENCY_MARKERS)


def emergency_notice(question):
    """Return a fixed notice for emergency wording, or an empty string."""
    return EMERGENCY_NOTICE if looks_like_emergency(question) else ""


def has_referral(text):
    """True when the answer points the user to a real health professional."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in REFERRAL_MARKERS)


def apply_safety_notice(question, answer):
    """Prepend the emergency notice so safety does not rely on the model."""
    answer = (answer or "").strip()
    notice = emergency_notice(question)
    if not notice:
        return answer
    return "{0}\n\n{1}".format(notice, answer).strip()


def chat(question, history=None, base_url=None, timeout=TIMEOUT):
    """Send one question to the locally served Spark-X2.6-4B endpoint."""
    url = (base_url or BASE_URL).rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(build_payload(question, history)).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return apply_safety_notice(question, extract_text(body))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print("Model: {0}".format(MODEL_ID))
    print("Endpoint: {0}".format(BASE_URL))
    print("\nYou: {0}\n".format(question))
    try:
        answer = chat(question)
    except (urllib.error.URLError, OSError, ValueError) as error:
        print(
            "Could not reach the Spark endpoint ({0}). Start a local "
            "OpenAI-compatible server for {1} and set SPARK_BASE_URL.".format(
                error, MODEL_ID
            ),
            file=sys.stderr,
        )
        return 1
    print("Assistant: {0}".format(answer or "(empty response)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
