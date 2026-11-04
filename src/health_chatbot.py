#!/usr/bin/env python3
"""Minimal health-knowledge chatbot built on Spark-X2.6-4B.

The application answers general health-education questions in plain language
and refuses to diagnose, prescribe, or replace a clinician. It talks to any
OpenAI-compatible Spark-X2.6 endpoint (llama.cpp, Ollama, SGLang, vLLM, MLX
server ...), so the demo can run fully offline after the weights are
downloaded.

Usage:
    python src/health_chatbot.py                    # runs DEMO_QUESTION
    python src/health_chatbot.py "my hands feel numb"
    python src/health_chatbot.py --offline          # print the request only

Environment:
    SPARK_BASE_URL   default http://127.0.0.1:30000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = (
    "You are a patient-education assistant powered by Spark-X2.6-4B. "
    "Answer in the user's language, in short plain sentences at roughly a "
    "grade-6 reading level. Explain general health information only: what a "
    "symptom can commonly mean, when to seek care, and self-care that is safe "
    "for most adults. Never give a diagnosis, never name prescription "
    "medicines or dosages, never tell the user to stop or change prescribed "
    "treatment, and never claim to replace a clinician. If the question "
    "suggests an emergency (chest pain, trouble breathing, stroke signs, "
    "severe bleeding, thoughts of self-harm, pregnancy complications), tell "
    "the user to contact local emergency services or a clinician right now. "
    "Do not ask for names, ID numbers, addresses or other personal data, and "
    "remind the user to confirm anything important with a qualified health "
    "worker."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since yesterday I have a dry "
    "cough, a 38.2 C fever and feel very tired. What could this be and when "
    "should I see a doctor?"
)

SAFETY_NOTE = (
    "Educational information only - not a diagnosis or prescription. "
    "For anything urgent, contact local emergency services."
)


def build_messages(question: str, history=None):
    """Build the chat message list sent to Spark-X2.6-4B.

    ``history`` is an optional list of ``{"role", "content"}`` dicts produced
    by earlier turns; it is inserted between the system prompt and the new
    question so the model keeps the conversation context.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError("history entries need role user/assistant and string content")
        if content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question: str, history=None, model: str = MODEL_ID,
                  max_tokens: int = MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body."""
    return {
        "model": model,
        "messages": build_messages(question, history),
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }


def parse_response(body):
    """Extract the assistant answer and fail loudly on an empty/shape error."""
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response shape from the Spark endpoint") from exc

    content = (message.get("content") or "").strip()
    if not content:
        raise ValueError(
            "Spark returned an empty answer; retry with a larger token budget"
        )
    return content


def ask(question: str, history=None, model: str = MODEL_ID,
        base_url: str = None, timeout: float = 120.0):
    """Send one question to the local Spark-X2.6-4B server and return the answer."""
    payload = build_payload(question, history, model=model)
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return parse_response(body)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health-knowledge chatbot")
    parser.add_argument("question", nargs="*", help="question to ask; defaults to DEMO_QUESTION")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="OpenAI-compatible Spark endpoint (default: %(default)s)")
    parser.add_argument("--offline", action="store_true",
                        help="print the request that would be sent, without calling the model")
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip() or DEMO_QUESTION

    if args.offline:
        print(json.dumps(build_payload(question), indent=2, ensure_ascii=False))
        print("\n" + SAFETY_NOTE, file=sys.stderr)
        return 0

    try:
        answer = ask(question, base_url=args.base_url)
    except (ValueError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        print("hint: start a local Spark-X2.6-4B server or set SPARK_BASE_URL",
              file=sys.stderr)
        return 1

    print(answer)
    print("\n" + SAFETY_NOTE, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
