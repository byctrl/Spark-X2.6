"""Minimal health-information chatbot built on Spark-X2.6-4B.

HER Hack-Astron #7 (Spark-X2.6 Open Source AI for Good Challenge).

The module is intentionally dependency-free: it talks to a local
OpenAI-compatible endpoint (llama.cpp / SGLang / vLLM) that serves
``XHToken/Spark-X2.6-4B``. The application is patient *education*, not
diagnosis: the system prompt forbids diagnoses and prescriptions and routes
every answer back to a qualified clinician.

Usage:
    python src/health_chatbot.py --offline   # print the prompt, no network
    python src/health_chatbot.py             # call the local endpoint

Environment:
    SPARK_BASE_URL  default http://127.0.0.1:30000/v1
    SPARK_API_KEY   default EMPTY (local servers ignore it)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")
REQUEST_TIMEOUT = 120

DISCLAIMER = (
    "This prototype provides general health education only. It is not "
    "medical advice, diagnosis or treatment. In an emergency, contact your "
    "local emergency services immediately."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-health helpline.\n"
    "Follow these rules in every reply:\n"
    "1. Use plain language at roughly a 6th-grade reading level: short sentences, short paragraphs, no jargon unless you explain it.\n"
    "2. Never diagnose, never name a likely disease, never prescribe or dose a medicine, and never tell the user to stop a treatment.\n"
    "3. Describe in general terms what the reported symptoms can relate to, and list the questions a clinician will probably ask.\n"
    "4. Always end by telling the user to contact a qualified clinician or their local health service.\n"
    "5. If the message suggests an emergency (for example chest pain, trouble breathing, heavy bleeding, fainting, or thoughts of self-harm), say clearly that emergency help is needed now and tell the user to call local emergency services.\n"
    "6. Do not ask for or repeat names, addresses, ID numbers, or other identifying details.\n"
    "7. If information is missing, say what is missing instead of guessing.\n"
    "8. Answer in the language the user writes in."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and a fever that comes back every evening. I am 34 and otherwise "
    "healthy. What should I do next?"
)


def build_messages(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat-completion message list for one user question.

    The system prompt is always first and the new question is always last.
    Raises ``ValueError`` for empty questions or malformed history entries.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant"):
            raise ValueError("history role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("history content must be a non-empty string")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(
    messages: List[Dict[str, str]],
    model: str = MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Wrap messages in an OpenAI-compatible chat-completions payload."""
    if not messages:
        raise ValueError("messages must not be empty")
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
    }


def extract_text(response: Dict[str, Any]) -> str:
    """Pull the assistant text out of a chat-completions response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("malformed model response: missing choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def generate(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    base_url: Optional[str] = None,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    """Send one question to the local Spark-X2.6-4B server and return the text."""
    payload = build_payload(build_messages(question, history))
    url = (base_url or BASE_URL).rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return extract_text(json.loads(body))


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--offline" in args:
        print(json.dumps(build_messages(DEMO_QUESTION), indent=2, ensure_ascii=False))
        print()
        print(DISCLAIMER)
        return 0

    try:
        answer = generate(DEMO_QUESTION)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(
            "Could not get an answer from the Spark-X2.6-4B endpoint "
            "({0}): {1}".format(BASE_URL, exc),
            file=sys.stderr,
        )
        print(
            "Start a local OpenAI-compatible server, or run with --offline "
            "to inspect the prompt without a network call.",
            file=sys.stderr,
        )
        return 1

    print("Question: " + DEMO_QUESTION)
    print()
    print(answer)
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
