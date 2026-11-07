"""Minimal health-literacy chatbot demo built on Spark-X2.6-4B.

Reference application for the HER Hack-Astron #7 "AI for Good" challenge
(health + accessibility track).

The script talks to a locally hosted Spark-X2.6-4B served by any
OpenAI-compatible runtime (vLLM / SGLang / llama.cpp server / Spark-MLX-LLM)
and explains health information in plain language.

Safety boundary: the assistant never diagnoses, never prescribes, and always
points the user to a qualified clinician for decisions about their own care.
The model output is treated as patient education, not as medical advice.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
MAX_QUESTION_CHARS = 2000
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 120

DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")

SYSTEM_PROMPT = (
    "You are a patient-education assistant at a community health help desk.\n"
    "Explain health information in simple everyday language for a person with no "
    "medical training.\n"
    "Rules you must always follow:\n"
    "1. Do not diagnose, do not name a likely condition, and do not recommend "
    "medicines, dosages, or treatments.\n"
    "2. Explain what the described signs generally mean, what to ask a clinician, "
    "and which red-flag signs mean seeking care immediately.\n"
    "3. If the message suggests an emergency (chest pain, trouble breathing, heavy "
    "bleeding, fainting, a very high fever in a child), tell the person to contact "
    "emergency services right away.\n"
    "4. Never ask for or repeat names, ID numbers, addresses, or other identifying "
    "details.\n"
    "5. Be calm and non-judgemental, and answer in the language the user writes in.\n"
    "Close every answer with: 'This is general information, not medical advice. "
    "Please confirm with a qualified health worker.'"
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough and a "
    "mild fever in the evening. I do not have a family doctor. What should I ask at "
    "the clinic, and when should I worry?"
)


def build_messages(question, history=None, max_chars=MAX_QUESTION_CHARS):
    """Build the chat messages sent to Spark-X2.6-4B for one question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError("history turns need role user/assistant and text content")
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question.strip()[:max_chars]})
    return messages


def build_payload(messages, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
    """Build an OpenAI-compatible chat-completions payload with a bounded budget."""
    if not messages:
        raise ValueError("messages must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    return {
        "model": MODEL_ID,
        "messages": list(messages),
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "stream": False,
    }


def parse_completion(data):
    """Extract the answer text from a chat-completions response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected completion payload: %r" % (data,)) from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def ask(question, base_url=None, history=None, timeout=REQUEST_TIMEOUT):
    """Send one question to the local Spark-X2.6-4B endpoint and return the answer."""
    payload = build_payload(build_messages(question, history=history))
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Spark-X2.6-4B at %s. Start an OpenAI-compatible "
            "server (see README#quickstart) or set SPARK_BASE_URL. Details: %s"
            % (url, exc)
        ) from exc
    return parse_completion(json.loads(body))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print("Question: %s\n" % question)
    try:
        print(ask(question))
    except (RuntimeError, ValueError) as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
