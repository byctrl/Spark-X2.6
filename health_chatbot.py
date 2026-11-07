#!/usr/bin/env python3
"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

A plain-language patient-education assistant for people who have limited
access to health information. It explains general health knowledge, tells the
user when to seek care, and always defers diagnosis and treatment to a
qualified clinician. It never prescribes medicine.

The application talks to a locally hosted Spark-X2.6-4B through any
OpenAI-compatible endpoint (vLLM, SGLang, llama.cpp server, Ollama, ...).
Only the Python standard library is required.

Start the model server first, for example:

    python -m sglang.launch_server --model-path XHToken/Spark-X2.6-4B --port 30000

Then run:

    python health_chatbot.py                       # built-in demo question
    python health_chatbot.py "Apa itu demam denggi?"
    python health_chatbot.py -i                    # interactive chat

Environment variables:
    SPARK_BASE_URL  OpenAI-compatible base URL (default http://127.0.0.1:30000/v1)
    SPARK_API_KEY   API key if the server requires one (default EMPTY)
    SPARK_TIMEOUT   request timeout in seconds (default 120)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")
TIMEOUT = float(os.environ.get("SPARK_TIMEOUT", "120"))

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information service. Reply in the same language as the user, using short "
    "plain-language sentences that a secondary-school reader can follow.\n"
    "Rules:\n"
    "1. Explain general health knowledge only: what a symptom commonly means, "
    "when to seek care, what to prepare for a clinic visit, how to prevent "
    "common illness.\n"
    "2. Never diagnose the user, never name a specific medicine or dose, and "
    "never tell the user to start, stop or change prescribed treatment.\n"
    "3. Always remind the user to confirm with a qualified clinician or the "
    "local health service, and mention the red-flag signs that need urgent care.\n"
    "4. If the question is not about health, say so briefly and offer a health "
    "topic you can explain.\n"
    "5. You cannot see images, lab results or prescriptions; ask for a short "
    "text description instead."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak tiga hari lalu, selepas balik "
    "dari perjalanan. Apa yang patut saya perhatikan?"
)

EMERGENCY_HINTS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "shortness of breath",
    "unconscious",
    "severe bleeding",
    "suicide",
    "overdose",
    "seizure",
    "nyeri dada",
    "sesak nafas",
    "tidak sedarkan diri",
    "pendarahan",
    "胸痛",
    "呼吸困难",
    "昏迷",
    "自杀",
)

EMERGENCY_NOTICE = (
    "IMPORTANT: if this may be an emergency (for example chest pain, trouble "
    "breathing, heavy bleeding, fainting, or thoughts of self-harm), contact "
    "your local emergency number or the nearest emergency department right now "
    "instead of waiting for this chat."
)


def needs_emergency_notice(question: str) -> bool:
    """Return True when the question mentions a possible emergency."""
    lowered = (question or "").lower()
    return any(hint in lowered for hint in EMERGENCY_HINTS)


def build_messages(question: str, history=None):
    """Build the chat-completions message list for one question."""
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


def chat(messages, base_url: str = BASE_URL, model: str = MODEL_ID,
         max_tokens: int = MAX_NEW_TOKENS, timeout: float = TIMEOUT) -> str:
    """Send messages to the Spark-X2.6-4B OpenAI-compatible endpoint."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("model returned no choices: " + json.dumps(body)[:500])
    return (choices[0].get("message", {}).get("content") or "").strip()


def ask_health_question(question: str, history=None, chat_fn=None) -> str:
    """Answer one health question, adding an emergency notice when relevant."""
    chat_fn = chat_fn or chat
    answer = chat_fn(build_messages(question, history))
    if needs_emergency_notice(question):
        return EMERGENCY_NOTICE + "\n\n" + answer
    return answer


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] in ("-i", "--interactive"):
            print("Health information assistant (Spark-X2.6-4B).")
            print("Type a question, or Ctrl-D to quit. Not a substitute for a doctor.\n")
            history = []
            while True:
                try:
                    question = input("> ").strip()
                except EOFError:
                    print()
                    return 0
                if not question:
                    continue
                answer = ask_health_question(question, history)
                print("\n" + answer + "\n")
                history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ])
        question = " ".join(argv) if argv else DEMO_QUESTION
        print("Question: " + question + "\n")
        print(ask_health_question(question))
        return 0
    except urllib.error.URLError as exc:
        print("Could not reach Spark-X2.6-4B at %s (%s)." % (BASE_URL, exc),
              file=sys.stderr)
        print("Start a local OpenAI-compatible server and retry.", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as exc:
        print("Request failed: %s" % (exc,), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
