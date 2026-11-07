"""Minimal health-information chatbot built on Spark-X2.6-4B.

HER Hack-Astron #7 (AI for Good) minimal submission.

What the model does here: it converts a lay person's health question into a
short plain-language education answer. Spark-X2.6-4B is reached through an
OpenAI-compatible local endpoint (llama.cpp, vLLM, SGLang or Spark-MLX-LLM),
so this file only needs the Python standard library.

The application never diagnoses and never prescribes: the system prompt keeps
the model inside patient education and points the reader to a qualified
clinician.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people with limited access to "
    "health information. Answer in plain language at roughly a 6th-grade "
    "reading level, in the same language as the question.\n"
    "Rules:\n"
    "1. Never give a diagnosis and never name a treatment, drug or dose.\n"
    "2. Explain what the described signs can generally mean, and name the "
    "warning signs that need urgent care.\n"
    "3. Always end by telling the reader to confirm with a qualified clinician "
    "or the local health service.\n"
    "4. Do not ask for or repeat personal identifiers.\n"
    "5. If the message describes an emergency, say to contact emergency "
    "services right now."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago. I have had a dry cough and fever "
    "for four days, and now I feel short of breath when I climb stairs. "
    "I have no money for a doctor right now. What should I know?"
)

DISCLAIMER = "This is general health information, not medical advice."


def build_messages(question, history=None):
    """Build the chat payload: system prompt, past turns, then the question."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": (question or "").strip()})
    return messages


def chat_completions_url(base_url):
    return base_url.rstrip("/") + "/chat/completions"


def extract_answer(body):
    """Pull the assistant text out of an OpenAI-compatible response body."""
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected response shape from the Spark endpoint")
    text = (content or "").strip()
    if not text:
        raise ValueError("Spark-X2.6-4B returned an empty answer")
    return text


def with_disclaimer(answer):
    """Append the fixed disclaimer exactly once."""
    text = (answer or "").strip()
    if DISCLAIMER.lower() in text.lower():
        return text
    return text + "\n\n" + DISCLAIMER


def ask_spark(question, history=None, base_url=None, api_key=None,
              timeout=DEFAULT_TIMEOUT):
    """Send one question to a locally served Spark-X2.6-4B and return text."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    base_url = base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL
    key = api_key or os.environ.get("SPARK_API_KEY")

    payload = {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.3,
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key

    request = urllib.request.Request(
        chat_completions_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return extract_answer(body)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    try:
        answer = with_disclaimer(ask_spark(question))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print("Spark-X2.6-4B is not reachable: %s" % exc, file=sys.stderr)
        print(
            "Start a local OpenAI-compatible server for XHToken/Spark-X2.6-4B "
            "and set SPARK_BASE_URL (default %s)." % DEFAULT_BASE_URL,
            file=sys.stderr,
        )
        return 1
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
