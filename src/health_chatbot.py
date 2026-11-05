"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Runs against a locally hosted OpenAI-compatible server (SGLang / vLLM / llama.cpp /
Spark-MLX-LLM). It is an education aid only: it never diagnoses, never prescribes,
and never replaces a clinician.

Usage (from the repository root):

    python src/health_chatbot.py
    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py "my question"
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
REQUEST_TIMEOUT = 120

DISCLAIMER = (
    "General health information only - not medical advice. "
    "Contact a qualified clinician or your local emergency services about your own situation."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant in a public-interest health information service. "
    "Explain health topics in plain language (roughly a 6th-8th grade reading level) and use a short "
    "numbered list when it helps."
    "\n\nHard rules:"
    "\n1. Never diagnose. Never state a definitive condition for this person. Never prescribe medicines, "
    "doses, or treatments."
    "\n2. Do not replace a clinician. Always tell the reader to consult a qualified health professional "
    "about their own situation."
    "\n3. If the description suggests an emergency (for example chest pain, trouble breathing, fainting, "
    "heavy bleeding, confusion, stiff neck with fever, or symptoms in a newborn), say clearly that urgent "
    "in-person care is needed now and to contact local emergency services."
    "\n4. Ask for missing context (age, duration, severity, pregnancy, chronic conditions, medicines) "
    "instead of guessing."
    "\n5. Be inclusive: make no assumptions about gender, disability, income, or country; offer accessible "
    "wording and say when a symptom may need a carer or an interpreter."
    "\n6. State that this is general information, not medical advice, and that local guidance may differ."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago. Since yesterday I have a dry cough, "
    "a fever of 38.4 C, and I feel very tired. I am 29 and not pregnant. "
    "What should I know and when should I see a doctor?"
)


def build_messages(
    question: str, history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = (turn or {}).get("role")
        content = ((turn or {}).get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_payload(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, object]:
    """Build the OpenAI-compatible request body for one turn."""
    return {
        "model": model,
        "messages": build_messages(question, history),
        "max_tokens": int(max_tokens),
        "temperature": 0.3,
        "stream": False,
    }


def extract_answer(body: Dict[str, object]) -> str:
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        content = body["choices"][0]["message"]["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response from the model server: {body!r}") from exc

    content = (content or "").strip()
    if not content:
        raise RuntimeError("The model returned an empty answer.")
    return content


def ask(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    """Send one question to a local Spark-X2.6-4B server and return the answer text."""
    payload = build_payload(question, history, model=model, max_tokens=max_tokens)
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach an OpenAI-compatible server at {url}: {exc}. "
            "Start Spark-X2.6-4B (see README Quickstart) or set SPARK_BASE_URL."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"The server at {url} did not return JSON.") from exc

    return extract_answer(body)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a general health question (education only)."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEMO_QUESTION,
        help="question text; defaults to the bundled demo scenario",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible endpoint (default: %(default)s)",
    )
    parser.add_argument("--model", default=MODEL_ID, help="model name (default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    print(DISCLAIMER)
    print()
    print(f"Q: {args.question}")
    print()
    try:
        answer = ask(
            args.question,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    print(answer)
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
