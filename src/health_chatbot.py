#!/usr/bin/env python3
"""Minimal health-information chatbot built on Spark-X2.6-4B.

Public-interest (health) demo for the HER Hack-Astron #7 challenge: a
plain-language health *education* assistant. It does not diagnose, prescribe
or replace a clinician, and emergency signals are surfaced before the model
is ever called.

Run:
    python -m src.health_chatbot --demo
    python -m src.health_chatbot "I have a cough and fever after travelling"

Generation requires (prompt helpers do not):
    pip install transformers torch
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = """You are a patient-education assistant for people who may have limited health literacy or limited access to care.

Rules you must always follow:
1. Use plain language, short sentences, and define any medical term you use.
2. Do NOT diagnose, do NOT name a likely disease, and do NOT prescribe or dose any medicine.
3. Do not claim certainty. Separate general knowledge from what depends on a clinician's assessment.
4. Give practical, low-risk next steps: what to observe, what to record, when and where to seek care.
5. If the message suggests an emergency (for example chest pain, trouble breathing, heavy bleeding, fainting, signs of stroke, severe pain in pregnancy, or thoughts of self-harm), tell the person to contact local emergency services or the nearest emergency department now.
6. Never ask for or repeat identifying details. Do not give legal, insurance or immigration advice.
7. Close by reminding the person that this is general information, not medical care.
"""

DEMO_QUESTION = (
    "I have had a cough and a mild fever for three days, and I got back "
    "from a trip last week. What should I pay attention to, and when "
    "should I see someone?"
)

EMERGENCY_SIGNS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "shortness of breath",
    "heavy bleeding",
    "fainted",
    "passed out",
    "stroke",
    "face drooping",
    "slurred speech",
    "seizure",
    "suicidal",
    "kill myself",
    "self-harm",
    "bleeding in pregnancy",
)

EMERGENCY_NOTICE = (
    "This may be an emergency. Please contact your local emergency number "
    "or go to the nearest emergency department now. This demo cannot help "
    "with emergencies."
)

DISCLAIMER = (
    "General health information only - not diagnosis, prescription or "
    "medical care. Always follow up with a qualified clinician."
)

_PIPELINE = None


def needs_emergency_notice(text: str) -> bool:
    """True when the text contains a red-flag emergency phrase."""
    lowered = (text or "").lower()
    return any(sign in lowered for sign in EMERGENCY_SIGNS)


def build_messages(
    question: str,
    history: Optional[Iterable[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt is always first, the current question always last, and
    emergency questions get an extra system reminder before the user turn.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    if needs_emergency_notice(question):
        messages.append({"role": "system", "content": EMERGENCY_NOTICE})

    messages.append({"role": "user", "content": question})
    return messages


def build_prompt(messages: Iterable[Dict[str, str]]) -> str:
    """Plain-text fallback prompt for runtimes without a chat template."""
    lines = ["<|{0}|>\n{1}".format(m["role"], m["content"]) for m in messages]
    lines.append("<|assistant|>")
    return "\n".join(lines)


def _get_pipeline(model_id: str = MODEL_ID):
    """Load the text-generation pipeline once, on first use."""
    global _PIPELINE
    if _PIPELINE is None:
        from transformers import pipeline  # imported lazily: tests stay light

        _PIPELINE = pipeline("text-generation", model=model_id, device_map="auto")
    return _PIPELINE


def generate(
    question: str,
    history: Optional[Iterable[Dict[str, str]]] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    model_id: str = MODEL_ID,
) -> str:
    """Answer a health question with Spark-X2.6-4B.

    Emergency questions are answered deterministically and never reach the
    model, so a bad sample cannot talk someone out of seeking urgent care.
    """
    if needs_emergency_notice(question):
        return EMERGENCY_NOTICE

    messages = build_messages(question, history)
    pipe = _get_pipeline(model_id)
    result = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )

    first = result[0]["generated_text"] if isinstance(result, list) else result
    if isinstance(first, list):
        return str(first[-1].get("content", "")).strip()
    return str(first).strip()


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="*", help="health question to ask")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the built-in demo question",
    )
    parser.add_argument("--model", default=MODEL_ID, help="model id or local path")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="generation budget",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    question = DEMO_QUESTION if args.demo or not args.question else " ".join(args.question)

    print("Model: {0}".format(args.model))
    print("Question: {0}".format(question))
    print("-" * 60)
    print(
        generate(
            question,
            max_new_tokens=args.max_new_tokens,
            model_id=args.model,
        )
    )
    print("-" * 60)
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
