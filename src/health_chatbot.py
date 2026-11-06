"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7 demo).

A patient-education assistant: it explains health topics in plain language.
It does not diagnose, prescribe, or replace a qualified clinician, and it
refuses to handle emergencies except to point at local emergency services.

The model is called through an OpenAI-compatible chat-completions endpoint
(Spark-MLX-LLM, SGLang, vLLM, llama.cpp server, ...). No third-party
dependencies: only the Python standard library.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health service. "
    "Explain health topics in plain language (roughly a 6th-grade reading level), "
    "in short paragraphs, and define any medical term you have to use. "
    "You do NOT diagnose, prescribe, or give dosages, and you never tell a person "
    "to start, stop, or change a prescribed treatment. Always say that this is "
    "general education, not medical advice, and advise seeing a qualified "
    "clinician about their own situation. "
    "If the message suggests an emergency (chest pain, difficulty breathing, "
    "heavy bleeding, fainting, thoughts of self-harm, complications in pregnancy "
    "or with a newborn), stop explaining and tell them to contact local emergency "
    "services or a crisis line right now. "
    "Do not ask for names, addresses, ID numbers, or other identifying details, "
    "and remind the user not to share them."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough and "
    "a mild fever in the evening. What could it be and what should I do?"
)

DISCLAIMER = (
    "General health education only - not medical advice. "
    "For your own situation, contact a qualified clinician."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    ``history`` is an optional list of earlier turns; only well-formed
    ``user``/``assistant`` turns are kept so a caller cannot inject a new
    system prompt. The returned list is new and never mutates the input.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})

    clean = (question or "").strip()
    if not clean:
        raise ValueError("question must not be empty")
    messages.append({"role": "user", "content": clean})
    return messages


def build_payload(messages, max_new_tokens=MAX_NEW_TOKENS, temperature=0.3):
    """Build the OpenAI-compatible request body for the Spark endpoint."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": False,
    }


def parse_response(payload):
    """Extract the assistant text from a chat-completions response body."""
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response shape from the Spark endpoint") from exc

    text = (content or "").strip()
    if not text:
        raise ValueError("the model returned an empty answer")
    return text


def generate(question, history=None, base_url=None, max_new_tokens=MAX_NEW_TOKENS):
    """Ask Spark-X2.6-4B for a plain-language answer to ``question``."""
    messages = build_messages(question, history)
    payload = build_payload(messages, max_new_tokens=max_new_tokens)
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "could not reach %s - start a local OpenAI-compatible server for %s "
            "and set SPARK_BASE_URL" % (url, MODEL_ID)
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError("the Spark endpoint returned invalid JSON") from exc

    return parse_response(body)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = argv[0] if argv else DEMO_QUESTION
    print("Model: %s" % MODEL_ID)
    print("Question: %s" % question)
    print("-" * 60)
    print(generate(question))
    print("-" * 60)
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
