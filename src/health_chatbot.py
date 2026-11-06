"""health_chatbot.py — minimal Spark-X2.6-4B health-information chatbot.

Public-interest demo for the HER Hack-Astron #7 challenge: inclusive, plain-language
access to general health information.

The application calls a locally hosted Spark-X2.6-4B model through its
OpenAI-compatible endpoint (SGLang / vLLM / Spark-MLX sever). It is education only:
it never diagnoses, prescribes or triages, and it points people to a qualified
clinician or emergency service for personal decisions.

Usage:
    python src/health_chatbot.py                # interactive demo
    python src/health_chatbot.py --demo         # run the demo question once
    python src/health_chatbot.py -q "..."       # ask a single question

Environment:
    SPARK_BASE_URL  default http://127.0.0.1:30000/v1
    SPARK_MODEL     default XHToken/Spark-X2.6-4B
    SPARK_API_KEY   optional bearer token
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health information service. "
    "Explain health topics in plain language (around a grade 6-8 reading level) and be inclusive "
    "of readers with low literacy, disabilities or limited access to care. "
    "Rules: never diagnose, prescribe, change medication or give dosages; never claim to replace a "
    "clinician; never ask for or repeat personal identifiers; say clearly when information is "
    "general and may not apply to the reader; always finish by recommending a qualified clinician "
    "or local health service for personal decisions. If the message suggests an emergency, tell "
    "the reader to contact emergency services right now. Answer in the language the user writes in. "
    "Keep the answer under 250 words."
)

DISCLAIMER = (
    "General health information only - not medical advice. "
    "For your own situation, please consult a qualified clinician or your local health service."
)

EMERGENCY_MESSAGE = (
    "This may be an emergency. Please contact your local emergency number or go to the nearest "
    "emergency department right now. This chatbot cannot assess emergencies, and waiting for a "
    "reply here is not safe."
)

# Simple guardrail: obvious emergency phrasing is answered without calling the model.
EMERGENCY_SIGNALS = (
    "chest pain",
    "nyeri dada",
    "cannot breathe",
    "can't breathe",
    "tidak bisa bernapas",
    "sesak napas berat",
    "heavy bleeding",
    "pendarahan hebat",
    "unconscious",
    "tidak sadar",
    "seizure",
    "kejang",
    "suicide",
    "bunuh diri",
    "overdose",
)

DEMO_QUESTION = (
    "I have had a dry cough and a fever of 38.2 C for three days, after returning from a trip. "
    "What should I pay attention to, and when should I see a doctor?"
)


def build_messages(
    question: str, history: Optional[Sequence[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Build the chat message list sent to Spark-X2.6-4B."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must not be empty")
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        role = str(turn.get("role", "")).strip()
        content = str(turn.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})
    return messages


def has_emergency_signal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(signal in lowered for signal in EMERGENCY_SIGNALS)


def format_reply(reply: str) -> str:
    """Attach the safety disclaimer to a model answer."""
    body = (reply or "").strip()
    if not body:
        body = "I could not produce an answer for this question. Please ask a clinician or local health service."
    if DISCLAIMER in body:
        return body
    return f"{body}\n\n{DISCLAIMER}"


class SparkClient:
    """Thin OpenAI-compatible client for a locally hosted Spark-X2.6-4B."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        self.base_url = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("SPARK_MODEL") or MODEL_ID
        self.api_key = api_key if api_key is not None else os.environ.get("SPARK_API_KEY", "")
        self.timeout = timeout

    def chat(self, messages: Sequence[Dict[str, str]], max_tokens: int = MAX_NEW_TOKENS) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach Spark-X2.6 endpoint at {self.base_url} (is the model server running?): {exc}"
            ) from exc
        except ValueError as exc:
            raise RuntimeError("Spark-X2.6 endpoint returned invalid JSON") from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected response from Spark-X2.6: {body!r}") from exc
        return (content or "").strip()


def answer(question: str, client, history: Optional[Sequence[Dict[str, str]]] = None) -> str:
    """Answer a health question with safety guardrails around the model call."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must not be empty")
    if has_emergency_signal(text):
        return EMERGENCY_MESSAGE
    reply = client.chat(build_messages(text, history), max_tokens=MAX_NEW_TOKENS)
    return format_reply(reply)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health information chatbot (education only).")
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo question")
    args = parser.parse_args(argv)

    client = SparkClient()

    if args.question:
        print(answer(args.question, client))
        return 0

    question = DEMO_QUESTION if args.demo else None
    if question is None:
        print("Spark-X2.6-4B health information chatbot. Type 'exit' to quit.\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not question or question.lower() in {"exit", "quit"}:
                return 0
            try:
                print(f"\nAssistant: {answer(question, client)}\n")
            except (RuntimeError, ValueError) as exc:
                print(f"\nError: {exc}\n")
    else:
        print(f"Question: {question}\n")
        print(f"Assistant: {answer(question, client)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
