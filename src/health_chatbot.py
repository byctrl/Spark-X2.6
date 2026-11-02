"""Spark Health Guide - an accessible health-information chatbot.

Minimal, dependency-free demo application for the HER Hack-Astron #7
"Spark-X2.6 Open Source AI for Good" challenge.

What the model does here: it turns a plain-language health question into a
short, non-diagnostic explainer, suggests questions the person can ask a real
clinician, and points to a concrete next step. The model is *not* a clinician:
it never diagnoses, never prescribes, and always escalates emergencies to
local services.

The module talks to any OpenAI-compatible endpoint serving
``XHToken/Spark-X2.6-4B`` (vLLM, SGLang, llama.cpp server, Ollama, ...).
No third-party packages are required - only the Python standard library.

Usage::

    python src/health_chatbot.py --question "I have a cough and a fever"
    python src/health_chatbot.py --base-url http://127.0.0.1:30000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "It cannot diagnose you. Please talk to a qualified clinician about your situation."
)

SYSTEM_PROMPT = (
    "You are Spark Health Guide, a patient-education assistant for people who "
    "may have limited access to health information.\n"
    "Write in plain language at roughly a 6th-grade reading level, in short paragraphs.\n"
    "You may: explain general health concepts, describe what a symptom usually means in "
    "general terms, suggest questions a person can ask a clinician, and explain when and "
    "where to seek care.\n"
    "You must not: diagnose, present a likely disease as a conclusion, prescribe or dose "
    "medication, interpret test results, or claim to replace professional care.\n"
    "If the person describes an emergency (chest pain, trouble breathing, heavy bleeding, "
    "fainting, seizure, stroke signs, or thoughts of self-harm), start your reply with "
    "'EMERGENCY:' and tell them to contact local emergency services right now.\n"
    "Never ask for names, addresses, ID numbers, or other identifying details.\n"
    "Always end with one line starting with 'Next step:' naming a concrete real-world "
    "action, such as calling a clinician, visiting a clinic, or contacting a local health line."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a dry cough and a mild "
    "fever in the evening. I do not have much money and I am not sure if I should "
    "see someone. What should I know before I decide?"
)

EMERGENCY_KEYWORDS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "trouble breathing",
    "heavy bleeding",
    "severe bleeding",
    "unconscious",
    "fainting",
    "seizure",
    "stroke",
    "suicide",
    "kill myself",
    "self-harm",
)

EMERGENCY_BANNER = (
    "EMERGENCY: if this is happening right now, contact your local emergency number "
    "(for example 112, 911, or 120) or go to the nearest emergency department immediately."
)


def is_emergency(text):
    """Return True when the text mentions a red-flag emergency symptom."""
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_payload(question, history=None, max_new_tokens=MAX_NEW_TOKENS, temperature=0.3):
    """Build an OpenAI-compatible request payload with bounded generation."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.9,
    }


def http_transport(base_url=DEFAULT_BASE_URL, api_key=None, timeout=120.0):
    """Return a transport that calls an OpenAI-compatible /chat/completions endpoint."""
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def _call(payload):
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        if api_key:
            request.add_header("Authorization", "Bearer " + api_key)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    return _call


class HealthChatbot:
    """Thin, safety-bounded wrapper around Spark-X2.6-4B.

    ``transport`` is a callable that takes the request payload and returns the
    raw model text. Injecting it keeps the app testable without a GPU.
    """

    def __init__(self, transport, max_new_tokens=MAX_NEW_TOKENS):
        if not callable(transport):
            raise TypeError("transport must be callable")
        self._transport = transport
        self.max_new_tokens = max_new_tokens

    def ask(self, question, history=None):
        payload = build_payload(question, history, max_new_tokens=self.max_new_tokens)
        raw = self._transport(payload)
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError("model returned an empty response")

        answer = raw.strip()
        emergency = is_emergency(question)
        if emergency and not answer.upper().startswith("EMERGENCY:"):
            answer = EMERGENCY_BANNER + "\n\n" + answer

        return {
            "question": question.strip(),
            "answer": answer,
            "emergency": emergency,
            "disclaimer": DISCLAIMER,
            "model": MODEL_ID,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask Spark-X2.6-4B a health question.")
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.environ.get("SPARK_API_KEY"))
    args = parser.parse_args(argv)

    bot = HealthChatbot(http_transport(args.base_url, args.api_key))
    result = bot.ask(args.question)
    print(result["answer"])
    print()
    print(result["disclaimer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
