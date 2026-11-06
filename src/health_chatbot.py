"""Minimal health-information chatbot built on Spark-X2.6-4B.

Purpose (HER Hack-Astron #7, AI for Good): give community health workers and
people with limited access to medical information a local, offline-capable way
to ask general health questions in plain language.

The model does the language work; this module only adds the safety boundary
(patient-education system prompt), a bounded request, strict response parsing
and a visible disclaimer. It never diagnoses or prescribes on its own.

Run against a local OpenAI-compatible Spark-X2.6-4B server::

    python src/health_chatbot.py "Saya demam tiga hari, apa yang harus saya lakukan?"

Environment:
    SPARK_BASE_URL  OpenAI-compatible base URL (default http://127.0.0.1:30000/v1)
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
DEFAULT_TIMEOUT = 120.0

SYSTEM_PROMPT = (
    "You are a patient-education assistant for community health workers and "
    "for people with limited access to medical information. Answer in the "
    "same language as the question, in plain language and short sentences. "
    "You explain general health information only: you never diagnose, you "
    "never prescribe, you never give medicine dosages, and you never tell "
    "someone to start, stop or change a treatment. Always say clearly when a "
    "symptom needs urgent in-person care (for example difficulty breathing, "
    "chest pain, fainting, heavy bleeding, or fever with confusion) and point "
    "to a local emergency service or clinician. When the question is about a "
    "specific person, recommend seeing a qualified clinician and state that "
    "your answer is general information, not medical advice."
)

DEMO_QUESTION = (
    "Saya batuk dan demam tiga hari setelah pulang dari perjalanan. "
    "Apa yang sebaiknya saya lakukan?"
)

DISCLAIMER = (
    "General health information only - not a diagnosis or a prescription. "
    "Please consult a qualified clinician or your local emergency service."
)


def build_messages(question, history=None):
    """Build the chat messages for one turn: system prompt, optional history, question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question, history=None, model=MODEL_ID):
    """Build a bounded OpenAI-compatible chat completion payload."""
    return {
        "model": model,
        "messages": build_messages(question, history),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
    }


def _post_json(url, payload, timeout=DEFAULT_TIMEOUT):
    """POST JSON with the standard library and return the decoded JSON body."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_answer(response):
    """Extract the assistant text from a completion response, or raise ValueError."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected completion response: missing choices[0].message") from exc
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def with_disclaimer(answer, disclaimer=DISCLAIMER):
    """Append the disclaimer unless the model already included it."""
    if disclaimer.strip().lower() in answer.lower():
        return answer
    return answer.rstrip() + "\n\n" + disclaimer


def chat(question, base_url=None, model=MODEL_ID, timeout=DEFAULT_TIMEOUT, history=None, transport=None):
    """Ask Spark-X2.6-4B one question and return the answer with a disclaimer."""
    base = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    payload = build_payload(question, history=history, model=model)
    post = transport or _post_json
    response = post(base + "/chat/completions", payload, timeout)
    return with_disclaimer(extract_answer(response))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Minimal Spark-X2.6-4B health-information chatbot (general information only)."
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=None, help=f"OpenAI-compatible base URL (default {DEFAULT_BASE_URL})")
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args(argv)

    try:
        answer = chat(args.question, base_url=args.base_url, model=args.model)
    except (urllib.error.URLError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "hint: start a local OpenAI-compatible Spark-X2.6-4B server "
            "and set SPARK_BASE_URL if it is not on " + DEFAULT_BASE_URL,
            file=sys.stderr,
        )
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
