"""Minimal Spark-X2.6-4B health-education chatbot.

Dependency-free (standard library only). It talks to any OpenAI-compatible
local server (vLLM / SGLang / llama.cpp / Spark-MLX-LLM) that serves
XHToken/Spark-X2.6-4B:

    python src/health_chatbot.py "Demam 3 hari, apa yang harus saya lakukan?"

Environment:
    SPARK_BASE_URL  base URL of the local model server
                    (default: http://127.0.0.1:30000/v1)

Safety: this is patient *education*, not medical advice. The system prompt
forbids diagnosis and prescription and asks the model to route people to a
licensed clinician or emergency service when needed. Model output is never
treated as ground truth by this script.
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
TIMEOUT_SECONDS = 120

DISCLAIMER = (
    "Educational information only. This tool does not diagnose or prescribe. "
    "For personal medical decisions contact a licensed clinician; in an "
    "emergency contact local emergency services."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health topics in plain "
    "language for people with low health literacy.\n"
    "Rules:\n"
    "1. Give general education only. Never diagnose, never prescribe, and "
    "never state a drug dose.\n"
    "2. State plainly when something needs professional care, and name the "
    "right kind of clinician or emergency service.\n"
    "3. Use short sentences and, where useful, a short bulleted list.\n"
    "4. If the question is unclear, ask one clarifying question.\n"
    "5. Reply in the language of the user's question."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam 38.5C sejak 3 hari lalu, dan baru pulang "
    "dari perjalanan. Apa yang sebaiknya saya lakukan?"
)


def build_messages(question, history=None):
    """Build the chat messages for one turn, keeping prior turns in order."""
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


def build_payload(question, history=None, max_new_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": max_new_tokens,
        "temperature": 0.3,
    }


def extract_reply(data):
    """Pull the assistant text out of a chat-completions response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected response shape from model server")
    content = (content or "").strip()
    if not content:
        raise ValueError("model returned an empty reply")
    return content


def endpoint_url(base_url=None):
    base = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL)
    return base.rstrip("/") + "/chat/completions"


def ask(question, history=None, base_url=None, max_new_tokens=MAX_NEW_TOKENS):
    """Send one question to the local Spark-X2.6-4B server and return the reply."""
    url = endpoint_url(base_url)
    body = json.dumps(build_payload(question, history, max_new_tokens)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "cannot reach a Spark-X2.6-4B server at {0}: {1}".format(url, exc)
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError("model server returned non-JSON output: {0}".format(exc))
    return extract_reply(payload)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print(DISCLAIMER, file=sys.stderr)
    print(ask(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
