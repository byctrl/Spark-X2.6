"""Minimal health-information chatbot demo for Spark-X2.6-4B.

Public-interest demo for HER Hack-Astron #7: an inclusive, plain-language
patient-education assistant. It gives general information only, never a
diagnosis or treatment plan, and points people to a clinician.

The model is served behind an OpenAI-compatible endpoint (vLLM, SGLang or
Spark-MLX-LLM). Configure it with the SPARK_BASE_URL environment variable.
"""

from __future__ import annotations

import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information service. Explain health topics in simple, respectful, "
    "plain language, and offer general self-care information that is safe "
    "for most adults.\n"
    "Rules you must follow:\n"
    "1. Do not give a diagnosis, prescription, dosage, or treatment plan.\n"
    "2. Do not claim to replace a clinician; always advise the person to "
    "contact a qualified clinician or local emergency service about their "
    "own situation.\n"
    "3. Say clearly when the information is general, and when urgent care "
    "may be needed (for example trouble breathing, chest pain, confusion, "
    "or a high fever that does not improve).\n"
    "4. Ask at most one short clarifying question if key details are "
    "missing, then give general guidance.\n"
    "5. Do not ask for names, addresses, or other identifying data.\n"
    "Answer in the language the person used."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam 38,5 derajat sejak 4 hari lalu, beberapa "
    "hari setelah pulang dari perjalanan. Apa yang sebaiknya saya lakukan?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages, max_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body for the local endpoint."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text from a chat-completion response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("response does not contain an assistant message")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("assistant message is empty")
    return content.strip()


def ask(question, history=None, base_url=None, timeout=REQUEST_TIMEOUT):
    """Send one question to the locally hosted Spark-X2.6-4B endpoint."""
    root = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = root + "/chat/completions"
    body = json.dumps(build_payload(build_messages(question, history))).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return parse_response(data)


def main():
    print("Demo question:", DEMO_QUESTION)
    try:
        print("\nSpark-X2.6-4B:")
        print(ask(DEMO_QUESTION))
    except Exception as error:  # noqa: BLE001 - demo helper, keep the message short
        print("Could not reach the local Spark endpoint:", error)
        print("Start an OpenAI-compatible server for", MODEL_ID)
        print("and set SPARK_BASE_URL (default %s)." % DEFAULT_BASE_URL)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
