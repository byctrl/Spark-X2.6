#!/usr/bin/env python3
"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

Public-interest demo: explain general health information in plain language for
people with limited access to a clinician. The model is never used to diagnose,
prescribe, or replace professional care.

Spark-X2.6-4B is a text-only model served behind an OpenAI-compatible API
(vLLM / SGLang / llama.cpp server). Point SPARK_BASE_URL at that server.

Usage:
    python src/health_chatbot.py "Saya demam tiga hari, apa yang harus saya perhatikan?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_TIMEOUT = 60.0

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health service.\n"
    "Your job is to explain general health information in plain, respectful language "
    "that a person without medical training can understand.\n"
    "Rules:\n"
    "1. Never diagnose, never name a disease as the person's condition, and never "
    "prescribe or change medication, dosage, or treatment.\n"
    "2. Explain what is generally known, which warning signs to watch for, and which "
    "kind of clinician or service to contact.\n"
    "3. If the message suggests an emergency, say clearly that the person should "
    "seek emergency care immediately.\n"
    "4. Use short sentences, define medical terms, and avoid jargon.\n"
    "5. Do not ask for or repeat identifying details such as full name, address, or "
    "ID numbers.\n"
    "6. State plainly that this is general information, not medical advice."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam sejak tiga hari setelah pulang dari luar kota. "
    "Apa yang sebaiknya saya perhatikan?"
)

DISCLAIMER = (
    "Ini informasi umum untuk edukasi, bukan diagnosis atau resep. "
    "Untuk keluhan yang menetap atau memburuk, hubungi dokter atau fasilitas "
    "kesehatan terdekat."
)

EMERGENCY_NOTE = (
    "Keluhan yang Anda sebut bisa merupakan tanda kegawatan. Segera hubungi "
    "layanan darurat atau pergi ke unit gawat darurat terdekat."
)

EMERGENCY_KEYWORDS = (
    "nyeri dada",
    "chest pain",
    "sesak napas",
    "shortness of breath",
    "tidak sadar",
    "unconscious",
    "pendarahan hebat",
    "bleeding heavily",
    "kejang",
    "seizure",
    "stroke",
    "bunuh diri",
    "suicide",
)


def has_emergency_signs(text):
    """Return True when the text mentions a red-flag symptom."""
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


def build_messages(question, history=None):
    """Build the OpenAI-style message list sent to Spark-X2.6-4B."""
    prompt = (question or "").strip()
    if not prompt:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = (turn or {}).get("role")
        content = ((turn or {}).get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})
    return messages


def _post_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def generate(messages, base_url=None, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS, timeout=DEFAULT_TIMEOUT, transport=None):
    """Call the local Spark-X2.6-4B endpoint and return the answer text."""
    limit = MAX_NEW_TOKENS if not max_tokens or max_tokens > MAX_NEW_TOKENS else int(max_tokens)
    root = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = root + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": limit,
        "temperature": 0.3,
    }

    sender = transport or _post_json
    data = sender(url, payload, timeout)

    try:
        text = (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("unexpected response from Spark endpoint: %r" % (data,))
    if not text:
        raise RuntimeError("Spark returned an empty answer")
    return text


def answer(question, history=None, base_url=None, max_tokens=MAX_NEW_TOKENS, transport=None):
    """Return the model explanation plus the mandatory safety notes."""
    messages = build_messages(question, history)
    body = generate(messages, base_url=base_url, max_tokens=max_tokens, transport=transport)
    note = DISCLAIMER
    if has_emergency_signs(question):
        note = EMERGENCY_NOTE + " " + DISCLAIMER
    return body + "\n\n" + note


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot (public-interest demo)"
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible Spark endpoint")
    args = parser.parse_args(argv)

    try:
        print(answer(args.question, base_url=args.base_url))
    except Exception as exc:  # surfaced to the operator, never hidden
        print("error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
