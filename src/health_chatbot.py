"""Minimal, safety-bounded health-information chatbot built on Spark-X2.6-4B.

Purpose: an inclusive plain-language health *education* assistant for people who
have limited access to trustworthy health information.

The bot never diagnoses, never prescribes and never replaces a clinician.
Application code routes red-flag symptoms to emergency care *before* the model
is called, so the model is never asked to triage an emergency.

Run against any OpenAI-compatible server that serves XHToken/Spark-X2.6-4B
(vLLM, SGLang, llama.cpp server, Spark-MLX-LLM):

    python src/health_chatbot.py --demo
    python src/health_chatbot.py "Batuk dan demam 3 hari, apa yang bisa saya lakukan?"

The default endpoint is http://127.0.0.1:30000/v1 and can be changed with
--base-url or the SPARK_BASE_URL environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_TIMEOUT = 120

DEMO_QUESTION = (
    "Saya pulang dari perjalanan seminggu lalu. Sejak dua hari lalu saya batuk "
    "kering dan demam 38 derajat. Saya tidak punya asuransi dan belum bisa ke "
    "dokter. Informasi apa yang perlu saya tahu, dan kapan saya harus mencari "
    "bantuan medis?"
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in simple, "
    "respectful language for a reader with no medical training. "
    "Rules you must always follow:\n"
    "1. You do NOT diagnose, you do NOT prescribe, and you do NOT give drug doses.\n"
    "2. Give general, well-established information (what a symptom can mean, "
    "self-care basics, how to prepare for a appointment, what to ask a clinician).\n"
    "3. Always state clearly when something needs a clinician, and say that this "
    "answer is education, not medical advice.\n"
    "4. If the user describes an emergency, tell them to contact local emergency "
    "services immediately.\n"
    "5. Never ask for or repeat identifying personal data. If information is "
    "missing, say what is missing instead of guessing.\n"
    "6. Answer in the language the user wrote in, at roughly a 6th-grade reading "
    "level, in short paragraphs or a short list."
)

# Coarse, deliberately small keyword net. It is a safety net, NOT a triage device
# and NOT a diagnosis; it only decides whether to hand the user an emergency
# message instead of a model answer.
RED_FLAG_TERMS = (
    "nyeri dada",
    "sakit dada",
    "chest pain",
    "sesak napas",
    "sulit bernapas",
    "shortness of breath",
    "can't breathe",
    "cannot breathe",
    "pendarahan hebat",
    "severe bleeding",
    "kehilangan kesadaran",
    "unconscious",
    "fainted",
    "wajah miring",
    "slurred speech",
    "bicara pelo",
    "kejang",
    "seizure",
    "ingin mengakhiri hidup",
    "suicidal",
    "kill myself",
)

URGENT_REPLY = (
    "This sounds like it may be an emergency. Please contact your local "
    "emergency number or go to the nearest emergency department now. "
    "If you can, ask someone to stay with you. "
    "This chatbot cannot assess emergencies and cannot give medical advice."
)

DISCLAIMER = (
    "\n\n---\nThis is general health education, not medical advice, diagnosis or "
    "treatment. Please confirm anything important with a qualified clinician."
)


def build_messages(question: str, history=None):
    """Build the OpenAI-style message list sent to Spark-X2.6-4B.

    ``history`` is an optional list of ``{"role", "content"}`` turns from
    earlier in the conversation; it is copied so callers are never mutated.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def needs_urgent_care(text: str) -> bool:
    """Return True when the text contains a coarse red-flag phrase."""
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(term in lowered for term in RED_FLAG_TERMS)


def format_answer(answer: str) -> str:
    """Normalise a model answer and append the education disclaimer."""
    cleaned = (answer or "").strip()
    if not cleaned:
        cleaned = (
            "The model returned an empty answer. Please try rephrasing your "
            "question, or ask a clinician or community health worker directly."
        )
    return cleaned + DISCLAIMER


def chat(question: str, history=None, base_url: str = DEFAULT_BASE_URL, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Call the local Spark-X2.6-4B OpenAI-compatible endpoint."""
    payload = {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def respond(question: str, history=None, client=chat) -> str:
    """Safety gate + model call + disclaimer.

    ``client`` is injectable so the flow can be tested without a model server.
    """
    if needs_urgent_care(question):
        return URGENT_REPLY
    return format_answer(client(question, history))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health-information chatbot")
    parser.add_argument("question", nargs="?", help="health question to ask")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo question")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SPARK_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL of the local Spark server",
    )
    args = parser.parse_args(argv)

    question = DEMO_QUESTION if args.demo else args.question
    if not question:
        parser.error("provide a question or use --demo")

    print("model:", MODEL_ID)
    print("question:", question)
    print()
    try:
        print(respond(question, base_url=args.base_url))
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
        print(
            "Could not reach a local Spark-X2.6-4B server at "
            f"{args.base_url} ({exc}). Start the server first, or pass --base-url.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
