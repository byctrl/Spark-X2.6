"""Minimal health-information chatbot built on Spark-X2.6-4B.

Public-interest demo for the HER Hack-Astron #7 "AI for Good" challenge:
a plain-language patient-education assistant that explains health topics
without diagnosing or prescribing.

The model is served by any OpenAI-compatible runtime (vLLM, SGLang, Ollama,
llama.cpp server, Spark-MLX-LLM). Point SPARK_BASE_URL at it::

    python src/health_chatbot.py
    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py

Only the Python standard library is used on purpose, so the demo runs on a
laptop without installing a model client SDK.
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
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health topics in plain "
    "language for readers with no medical training.\n"
    "Rules:\n"
    "1. Do not diagnose, do not present a likely disease as a conclusion, and "
    "do not recommend prescription drugs or dosages.\n"
    "2. Explain in general terms what the described signs can mean, which "
    "questions a clinician may ask, and which red-flag signs need urgent care.\n"
    "3. End every answer by advising the reader to consult a qualified "
    "clinician about their own situation.\n"
    "4. If information is missing, say what is missing instead of guessing.\n"
    "5. Reply in the language of the question."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago. For three days I have had a dry "
    "cough, a fever of about 38.5 C in the evening, and I feel very tired. I "
    "have no insurance and I am not sure whether to wait or see someone. "
    "What should I pay attention to?"
)


class ChatbotError(RuntimeError):
    """Raised when the Spark-X2.6 endpoint returns an unusable response."""


def build_messages(question, history=None):
    """Build the chat message list sent to Spark-X2.6-4B.

    The safety system prompt is always first and cannot be replaced by the
    caller. ``history`` may only contribute user/assistant turns.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question, history=None, max_new_tokens=MAX_NEW_TOKENS, temperature=0.2):
    """Build the OpenAI-compatible request body for one turn."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
    }


def chat(question, base_url=None, history=None, max_new_tokens=MAX_NEW_TOKENS, timeout=REQUEST_TIMEOUT):
    """Send one question to a local Spark-X2.6-4B server and return the reply."""
    payload = build_payload(question, history, max_new_tokens)
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatbotError("unexpected response from %s: %r" % (url, body)) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B for plain-language health information (not medical advice)."
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION, help="question to ask the model")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible Spark-X2.6 endpoint")
    args = parser.parse_args(argv)

    print("model:    %s" % MODEL_ID)
    print("endpoint: %s" % args.base_url)
    print("question: %s\n" % args.question)

    try:
        answer = chat(args.question, base_url=args.base_url)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(
            "error: cannot reach the Spark-X2.6 endpoint (%s).\n"
            "Start an OpenAI-compatible Spark-X2.6-4B server and set SPARK_BASE_URL." % exc,
            file=sys.stderr,
        )
        return 1
    except ChatbotError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1

    print(answer)
    print("\nThis is general information, not medical advice. Please consult a qualified clinician.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
