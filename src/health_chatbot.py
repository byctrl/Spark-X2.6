"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Scope and safety boundary
-------------------------
This is a *patient-education* helper, not a medical device. It explains health
topics in plain language and always redirects care decisions to a qualified
clinician. It must never diagnose, prescribe, or give dosages.

What the model does here
------------------------
``XHToken/Spark-X2.6-4B`` performs all of the natural-language work: it reads
the user's question together with the fixed system prompt and writes the plain
language explanation. Everything else in this file (prompt assembly, HTTP call,
response parsing) is deterministic glue code.

The model is served locally by any OpenAI-compatible runtime (vLLM, SGLang,
llama.cpp server, ...) so that no health data leaves the machine:

    # terminal 1 - serve the model locally
    python -m vllm.entrypoints.openai.api_server --model XHToken/Spark-X2.6-4B

    # terminal 2 - ask a question
    python -m src.health_chatbot

Environment variables
---------------------
SPARK_BASE_URL  OpenAI-compatible base URL (default http://127.0.0.1:30000/v1)
SPARK_API_KEY   API key for the local server (default "EMPTY", ignored locally)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"

SYSTEM_PROMPT = (
    "You are a patient-education assistant in a public health information service. "
    "Explain health topics in plain everyday language and, when helpful, in the "
    "language the user writes in. You must never break these rules:\n"
    "1. You cannot diagnose. Say so plainly instead of naming a condition as the cause.\n"
    "2. Do not prescribe, recommend dosages, or tell the user to start, stop or change "
    "any treatment or medicine.\n"
    "3. Always send the user to a qualified clinician or pharmacist for decisions about "
    "their own care, and mention the local emergency number when relevant.\n"
    "4. If the message suggests an emergency (chest pain, difficulty breathing, heavy "
    "bleeding, fainting, confusion, thoughts of self-harm), tell them to contact "
    "emergency services right now.\n"
    "5. Never ask for or repeat names, ID numbers, addresses or other personal "
    "identifiers, and never invent medical facts.\n"
    "Answer in short paragraphs and finish with two or three questions the user can "
    "ask their clinician."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days, after returning from a trip "
    "abroad. What should I know before I see a doctor?"
)


def build_messages(question: str) -> list[dict[str, str]]:
    """Return the chat messages sent to Spark-X2.6-4B for one question."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def build_payload(question: str) -> dict[str, object]:
    """Return the OpenAI-compatible request body for one question."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.3,
    }


def extract_answer(data: dict) -> str:
    """Pull the assistant text out of an OpenAI-compatible response body."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("unexpected model response: %r" % (data,)) from exc
    text = (content or "").strip()
    if not text:
        raise RuntimeError("model returned an empty answer")
    return text


def chat(question: str, base_url: str | None = None, timeout: float = 120.0) -> str:
    """Send one question to the locally served model and return its answer."""
    root = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    body = json.dumps(build_payload(question)).encode("utf-8")
    request = urllib.request.Request(root + "/chat/completions", data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", "Bearer " + os.environ.get("SPARK_API_KEY", "EMPTY"))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return extract_answer(json.loads(raw))


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    print("Q:", question)
    try:
        answer = chat(question)
    except Exception as exc:  # noqa: BLE001 - CLI keeps the failure readable
        print("error:", exc, file=sys.stderr)
        print("hint: start a local OpenAI-compatible server for " + MODEL_ID, file=sys.stderr)
        return 1
    print()
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
