"""Minimal plain-language health-information chatbot built on Spark-X2.6-4B.

Public-interest demo for HER Hack-Astron #7 ("AI for Good"). The assistant
answers general health-education questions in simple language. It does NOT
diagnose, prescribe, or replace a clinician: every answer ends with a
reminder to consult a qualified professional, and emergency wording is
surfaced when the user describes urgent symptoms.

The model is served locally behind an OpenAI-compatible endpoint (vLLM /
SGLang / llama.cpp / Ollama), so the same code runs fully offline.

Usage::

    python src/health_chatbot.py
    python src/health_chatbot.py --question "How do I take my blood pressure?"

Set ``SPARK_BASE_URL`` (default ``http://127.0.0.1:30000/v1``) to point at the
Spark-X2.6-4B server you started.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = (
    "You are a patient-education assistant powered by Spark-X2.6-4B. "
    "Explain health information in short, plain sentences that a reader with "
    "limited medical literacy or limited literacy can follow. "
    "Use at most one short list, and define any medical word you must use. "
    "You must NOT diagnose, prescribe, give dosages, or tell the reader to "
    "start or stop a medicine. You must NOT claim to replace a doctor or a "
    "nurse. Instead, explain what the symptom or document usually means, "
    "what questions to ask a clinician, and which warning signs need urgent "
    "care. If the reader describes a possible emergency (chest pain, trouble "
    "breathing, heavy bleeding, fainting, thoughts of self-harm, labour "
    "complications, or a very sick infant), first tell them to contact local "
    "emergency services or go to the nearest emergency department now. "
    "Never invent sources, statistics, or local phone numbers. "
    "If you are unsure, say so plainly and ask a short clarifying question."
)

DISCLAIMER = (
    "This is general health education, not medical advice, diagnosis, or a "
    "prescription. Please check with a qualified health worker about your own "
    "situation."
)

EMERGENCY_HINTS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "trouble breathing",
    "shortness of breath",
    "heavy bleeding",
    "unconscious",
    "fainted",
    "seizure",
    "suicidal",
    "kill myself",
    "self-harm",
    "bleeding in pregnancy",
    "no fetal movement",
    "baby will not feed",
)

EMERGENCY_NOTICE = (
    "This sounds like it could be an emergency. Please contact your local "
    "emergency number or go to the nearest emergency department now, and do "
    "not wait for an online answer."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and now I have a dry cough and a "
    "fever of 38.5 C that will not go away. What should I do?"
)


def build_messages(
    question: str, history: Optional[Sequence[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Return the chat messages sent to Spark-X2.6-4B for one question."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must not be empty")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})
    return messages


def is_emergency_wording(text: str) -> bool:
    """Cheap keyword screen; the model prompt still carries the real policy."""
    lowered = (text or "").lower()
    return any(hint in lowered for hint in EMERGENCY_HINTS)


def finalize_answer(answer: str, question: str) -> str:
    """Guarantee emergency signposting and the education disclaimer."""
    text = (answer or "").strip()
    if not text:
        text = (
            "I could not produce an answer for that question. Please try "
            "rephrasing it, or ask a qualified health worker directly."
        )
    if is_emergency_wording(question) and "emergency" not in text.lower():
        text = EMERGENCY_NOTICE + "\n\n" + text
    if DISCLAIMER not in text:
        text = text + "\n\n" + DISCLAIMER
    return text


@dataclass
class OpenAICompatibleBackend:
    """Talks to a locally served Spark-X2.6-4B ``/v1/chat/completions``."""

    base_url: str = DEFAULT_BASE_URL
    model: str = MODEL_ID
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float = 120.0

    def complete(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                "Unexpected response from %s: %r" % (self.base_url, body)
            ) from None
        return (content or "").strip()


@dataclass
class HealthChatbot:
    """Thin wrapper: build messages, call the model, enforce safety framing."""

    complete: Callable[[List[Dict[str, str]]], str]
    history: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_openai_compatible(
        cls,
        base_url: str = DEFAULT_BASE_URL,
        model: str = MODEL_ID,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = 120.0,
    ) -> "HealthChatbot":
        backend = OpenAICompatibleBackend(
            base_url=base_url,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        return cls(complete=backend.complete)

    def ask(self, question: str) -> str:
        messages = build_messages(question, self.history)
        raw = self.complete(messages)
        answer = finalize_answer(raw, question)
        self.history.append({"role": "user", "content": question.strip()})
        self.history.append({"role": "assistant", "content": answer})
        return answer


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plain-language health-information chatbot on Spark-X2.6-4B"
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    chatbot = HealthChatbot.from_openai_compatible(
        base_url=args.base_url, max_new_tokens=args.max_new_tokens
    )
    print(chatbot.ask(args.question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
