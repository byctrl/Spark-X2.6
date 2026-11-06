"""Minimal health-information chatbot powered by Spark-X2.6-4B.

Public-interest demo for HER Hack-Astron #7 (health information + accessibility).
The model is used only for patient education in plain language: it explains
health topics, points out warning signs and redirects the user to a qualified
clinician. It does not diagnose and does not prescribe.

The script talks to a locally hosted Spark-X2.6-4B OpenAI-compatible endpoint
(SGLang / vLLM / llama.cpp server / MLX server). Only the Python standard
library is used, so no extra dependencies are required.

Quick check (with a local server running on the default port):

    python src/health_chatbot.py

Configuration:
    SPARK_BASE_URL  default http://127.0.0.1:30000/v1
    SPARK_API_KEY   default EMPTY (local servers usually ignore it)
"""

import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
DEFAULT_API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant in a public-interest demo. "
    "Explain health topics in plain, simple language for readers with low medical "
    "literacy: short sentences, no jargon, and define any term you must use. "
    "You must NOT diagnose, prescribe, or recommend specific medicines or doses. "
    "Never claim to replace a clinician. "
    "Always list red-flag warning signs and tell the user to contact a qualified "
    "clinician, and to use emergency services immediately in an emergency. "
    "If the user may be in danger, say clearly that they should seek care now. "
    "Answer in the language the user writes in. Keep answers short."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam sejak tiga hari lalu, dan baru pulang dari "
    "perjalanan. Apa yang sebaiknya saya perhatikan?"
)

DISCLAIMER = (
    "This demo gives general health information only. It is not a diagnosis or a "
    "treatment plan and does not replace a qualified clinician."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    ``history`` is an optional list of previous turns, each a dict with a
    ``role`` of ``"user"`` or ``"assistant"`` and text ``content``.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role not in ("user", "assistant"):
            raise ValueError("history turns need role 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("history turns need non-empty text content")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages, model=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body."""
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.3,
        "stream": False,
    }


def _post_json(payload, base_url, api_key, timeout):
    """Default transport: POST to an OpenAI-compatible chat endpoint."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", "Bearer " + api_key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach the Spark-X2.6 endpoint at %s: %s" % (url, exc)
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Spark-X2.6 endpoint returned invalid JSON") from exc


def parse_answer(response):
    """Extract the assistant text from a chat-completions response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("unexpected response from the Spark-X2.6 endpoint") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Spark-X2.6 returned an empty answer")
    return content.strip()


def generate(question, history=None, base_url=None, api_key=None,
             max_new_tokens=MAX_NEW_TOKENS, timeout=REQUEST_TIMEOUT,
             transport=None):
    """Ask Spark-X2.6-4B one health question and return the plain-language answer."""
    send = transport or _post_json
    messages = build_messages(question, history)
    payload = build_payload(messages, max_new_tokens=max_new_tokens)
    response = send(
        payload,
        base_url or DEFAULT_BASE_URL,
        DEFAULT_API_KEY if api_key is None else api_key,
        timeout,
    )
    return parse_answer(response)


def format_reply(answer):
    """Answer text plus the fixed safety disclaimer."""
    return "%s\n\n%s" % (answer.strip(), DISCLAIMER)


def main(question=DEMO_QUESTION):
    answer = generate(question)
    print("Question: %s" % question)
    print()
    print("Answer: %s" % format_reply(answer))
    return answer


if __name__ == "__main__":
    import sys

    main(" ".join(sys.argv[1:]).strip() or DEMO_QUESTION)
