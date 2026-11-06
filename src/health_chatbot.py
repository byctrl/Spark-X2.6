"""Minimal health-information chatbot built on Spark-X2.6-4B.

HER Hack-Astron #7 / Spark-X2.6 Open Source AI for Good Challenge.

The chatbot answers general health-education questions in plain language for
people who cannot easily reach a clinician. It never diagnoses, never
prescribes, and escalates emergency red flags to a human clinician or the
local emergency service.

Model: https://huggingface.co/XHToken/Spark-X2.6-4B

Run the illustrated demo against a local OpenAI-compatible Spark server
(SGLang / vLLM / llama.cpp server / MLX server):

    python src/health_chatbot.py

Environment variables:
    SPARK_BASE_URL  default http://127.0.0.1:30000/v1
    SPARK_API_KEY   default "EMPTY"
    SPARK_MODEL     default XHToken/Spark-X2.6-4B
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Callable, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 180

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public health information service. "
    "Use plain language at roughly a 6th-grade reading level, short paragraphs and short sentences. "
    "You explain general health information only. "
    "Never diagnose, never tell the reader which disease they have, and never recommend "
    "prescription drugs or doses. "
    "Always tell the reader to confirm any decision with a qualified clinician. "
    "If the description suggests an emergency (chest pain, trouble breathing, heavy bleeding, "
    "fainting, sudden weakness, confusion, stiff neck with fever, or thoughts of self-harm), "
    "say clearly that this needs urgent in-person care and to contact local emergency services now. "
    "Do not ask for or repeat identifying details. "
    "Explain what the reader can watch for and what kind of clinician to contact. "
    "End with one short sentence reminding the reader that this is general information, "
    "not medical advice."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago and since yesterday I have a dry cough, "
    "a temperature of 38.2 C and I feel very tired. I take medicine for high blood pressure. "
    "What should I watch for and when should I see a doctor?"
)

DISCLAIMER = "This is general information, not medical advice."

# Red-flag phrases handled in application code, before any model call.
EMERGENCY_TERMS = (
    "chest pain",
    "crushing chest",
    "cannot breathe",
    "can't breathe",
    "trouble breathing",
    "heavy bleeding",
    "won't stop bleeding",
    "fainted",
    "unconscious",
    "seizure",
    "sudden weakness",
    "face drooping",
    "slurred speech",
    "stiff neck",
    "suicidal",
    "kill myself",
    "self-harm",
    "overdose",
)

EMERGENCY_REPLY = (
    "This sounds like it could be an emergency.\n\n"
    "Please contact your local emergency number or go to the nearest emergency "
    "department now. Do not wait for a written answer here.\n\n"
    "If you are with someone who is unconscious, has chest pain, or is having "
    "trouble breathing, call emergency services immediately.\n\n"
    + DISCLAIMER
)

ModelClient = Callable[[List[Dict[str, str]]], str]


def is_emergency(text: str) -> bool:
    """Return True when the message contains an urgent red-flag phrase."""
    lowered = (text or "").lower()
    return any(term in lowered for term in EMERGENCY_TERMS)


def build_messages(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the Spark chat messages for one health question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    if history:
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question.strip()})
    return messages


class SparkHTTPClient:
    """Tiny client for a locally hosted Spark-X2.6 OpenAI-compatible server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = MODEL_ID,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        self.base_url = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("SPARK_API_KEY") or "EMPTY"
        self.model = model or os.environ.get("SPARK_MODEL") or MODEL_ID
        self.timeout = timeout

    @property
    def endpoint(self) -> str:
        return self.base_url + "/chat/completions"

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        return self.chat(messages)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


def answer(
    question: str,
    client: Optional[ModelClient] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Answer one health question with Spark-X2.6-4B and a safety floor."""
    if is_emergency(question):
        return EMERGENCY_REPLY

    messages = build_messages(question, history=history)
    reply = (client or SparkHTTPClient())(messages)
    reply = (reply or "").strip()
    if not reply:
        reply = "I could not produce an answer. Please ask a clinician or a local health worker."
    if DISCLAIMER.lower() not in reply.lower():
        reply = reply + "\n\n" + DISCLAIMER
    return reply


def main() -> int:
    print("Spark-X2.6-4B health information chatbot")
    print("Model:", MODEL_ID)
    print("Endpoint:", SparkHTTPClient().endpoint)
    print()
    print("Demo question:")
    print(DEMO_QUESTION)
    print()
    try:
        print(answer(DEMO_QUESTION))
    except Exception as exc:  # pragma: no cover - demo only
        print("Could not reach the Spark server:", exc, file=sys.stderr)
        print("Start a local OpenAI-compatible server or set SPARK_BASE_URL.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
