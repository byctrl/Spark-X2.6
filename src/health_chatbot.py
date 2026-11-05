"""Minimal health-knowledge chatbot for the Spark-X2.6 AI-for-Good challenge.

The chatbot gives general health education in plain language and never
diagnoses, prescribes, or collects identifying details. It talks to a locally
hosted Spark-X2.6-4B OpenAI-compatible endpoint (vLLM / SGLang / llama.cpp).

Run:
    python src/health_chatbot.py --question "Batuk dan demam setelah perjalanan"
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant at a community health desk.\n"
    "Answer in simple, plain language and short paragraphs.\n"
    "You do NOT diagnose conditions, interpret test results, or recommend "
    "prescription drugs or dosages.\n"
    "For anything urgent or personal, tell the reader to contact a qualified "
    "clinician or the local emergency service.\n"
    "Never ask for or repeat identifying details such as names, ID numbers, "
    "addresses, or phone numbers.\n"
    "If the question is outside general health education, say so briefly and "
    "suggest who could help."
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "Please see a qualified clinician for your own situation."
)

DEMO_QUESTION = (
    "Saya batuk dan demam sejak tiga hari setelah perjalanan. "
    "Hal apa yang sebaiknya saya perhatikan?"
)


def build_messages(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat messages for one user question plus optional history."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(
    messages: List[Dict[str, str]],
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Build the OpenAI-compatible chat completion request body."""
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False,
    }


def extract_answer(response: Dict[str, Any]) -> str:
    """Pull the assistant text out of a chat completion response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("response does not contain a chat completion message") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")

    answer = content.strip()
    if DISCLAIMER.lower() not in answer.lower():
        answer = "%s\n\n%s" % (answer, DISCLAIMER)
    return answer


def ask(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    base_url: Optional[str] = None,
    timeout: float = 120.0,
) -> str:
    """Send one question to the local Spark-X2.6-4B endpoint and return the answer."""
    endpoint = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    payload = build_payload(build_messages(question, history))
    request = urllib.request.Request(
        url=endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return extract_answer(body)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the local Spark-X2.6-4B health-education chatbot a question."
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args(argv)

    try:
        print(ask(args.question, base_url=args.base_url))
    except (urllib.error.URLError, ValueError, OSError) as exc:
        print("error: %s" % exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
