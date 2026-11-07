#!/usr/bin/env python3
"""Minimal patient-education chatbot built on Spark-X2.6-4B.

This is the AI-for-Good demo for HER Hack-Astron #7 (health / accessibility).
It talks to a *locally hosted* OpenAI-compatible server that serves
``XHToken/Spark-X2.6-4B`` (llama.cpp, vLLM, SGLang, Spark-MLX-LLM, ...).
No third-party Python packages are required.

What the model does here:
    Given a plain-language health question, Spark-X2.6-4B writes a short
    patient-education answer: general information, what to note down before an
    appointment, questions to ask a clinician, and when to seek urgent care.
    The model never diagnoses or prescribes; a human clinician stays in charge.

Usage:
    python src/health_chatbot.py
    python src/health_chatbot.py --question "I have had a fever for 3 days"
    python src/health_chatbot.py --base-url http://127.0.0.1:30000/v1

Environment:
    SPARK_BASE_URL  default http://127.0.0.1:8080/v1
    SPARK_API_KEY   default sk-local (most local servers ignore it)
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
TEMPERATURE = 0.3
TIMEOUT_SECONDS = 180

DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:8080/v1")
DEFAULT_API_KEY = os.environ.get("SPARK_API_KEY", "sk-local")

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people with limited access to health information.\n"
    "Audience: the general public, including readers with low health literacy or limited\n"
    "access to a clinician who speaks their language.\n"
    "Your job: explain health topics in plain language and help the person prepare for a\n"
    "conversation with a qualified health professional.\n"
    "\n"
    "Hard rules:\n"
    "1. Do not diagnose. Do not tell the person which disease they probably have.\n"
    "2. Do not prescribe and do not recommend dosages. Do not suggest starting, stopping\n"
    "   or changing any medicine, including over-the-counter drugs and traditional remedies.\n"
    "3. Do not interpret lab results, scans or prescriptions the way a clinician would.\n"
    "4. Always tell the reader to confirm with a qualified health professional.\n"
    "5. Name red-flag symptoms that mean seeking care now (for example chest pain,\n"
    "   difficulty breathing, fainting, confusion, heavy bleeding, fever in a newborn, or\n"
    "   thoughts of self-harm) and say to contact local emergency services.\n"
    "6. Never ask for or repeat identifying details: names, addresses, phone numbers, ID\n"
    "   numbers or insurance numbers.\n"
    "7. If the request is not health education, say so briefly and redirect.\n"
    "\n"
    "Answer format:\n"
    "- What this may involve (general information, not a diagnosis)\n"
    "- What to note down before an appointment\n"
    "- Questions to ask a clinician\n"
    "- When to seek urgent care\n"
    "Keep it short, use simple sentences, and answer in the language the person used.\n"
    "End with one line: 'This is general information, not medical advice.'\n"
)

DEMO_QUESTION = (
    "I have had a dry cough and a fever around 38.5 C for three days. It started two days "
    "after I came back from a trip abroad. What should I know before I see a doctor?"
)


def build_messages(question, history=None):
    """Return the chat messages sent to Spark-X2.6-4B for one question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("history turns must be dicts")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError("history turns must be user/assistant with string content")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question, history=None, max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
    """Build the OpenAI-compatible request body."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }


def parse_reply(data):
    """Extract the answer text from an OpenAI-compatible response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response shape from the model server") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("the model returned an empty answer")
    return content.strip()


def ask(question, history=None, base_url=None, api_key=None, timeout=TIMEOUT_SECONDS,
        max_tokens=MAX_NEW_TOKENS):
    """Send one question to the local Spark-X2.6-4B server and return the answer."""
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    body = json.dumps(build_payload(question, history, max_tokens=max_tokens)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + (api_key or DEFAULT_API_KEY),
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError("model server returned HTTP %s: %s" % (exc.code, detail)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "could not reach %s (%s). Start a local OpenAI-compatible server for %s, "
            "for example: llama-server -m Spark-X2.6-4B-Q4_K_M.gguf --port 8080"
            % (url, exc.reason, MODEL_ID)
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("model server did not return JSON") from exc
    return parse_reply(data)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B patient-education chatbot (offline, local server)"
    )
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    print("model:    %s" % MODEL_ID)
    print("endpoint: %s" % args.base_url)
    print("")
    print("Q: %s" % args.question)
    print("")
    try:
        print(ask(args.question, base_url=args.base_url, max_tokens=args.max_tokens))
    except (RuntimeError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
