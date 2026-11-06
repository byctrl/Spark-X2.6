"""Minimal Spark-X2.6-4B health-information chatbot.

Public-interest demo for HER Hack-Astron #7: an inclusive, plain-language
health knowledge assistant. It educates; it never diagnoses or prescribes.

Run the built-in demo (downloads the model on first use):

    python src/health_chatbot.py --demo

Run an interactive chat:

    python src/health_chatbot.py

Or inject your own generator (tests, mock, another runtime) via
``answer(question, generate_fn=...)``.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Optional, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people with low health literacy. "
    "Use short sentences and plain language (around a 6th-grade reading level). "
    "Explain what a symptom or term usually means, what questions to ask a clinician, "
    "and when to seek care urgently. "
    "Rules: never give a diagnosis, never name a prescription medicine or dose, "
    "never tell the user to stop or change a treatment, and never claim to replace a "
    "clinician. If the situation sounds urgent, say clearly that the person should "
    "contact emergency services or a clinician now. State uncertainty instead of "
    "guessing. Answer in the language the user writes in."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a cough and a mild fever. "
    "I do not have a doctor here and I am not sure if this is serious. What should I do?"
)

URGENT_HINTS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "difficulty breathing",
    "severe bleeding",
    "unconscious",
    "suicidal",
    "seizure",
    "stiff neck",
)

EMERGENCY_LINE = "If this is an emergency, contact your local emergency number now."

DISCLAIMER = (
    "This assistant gives general health education only. It is not a doctor and "
    "cannot diagnose or prescribe. Always check with a qualified clinician."
)

FALLBACK_ANSWER = (
    "I could not produce an answer. Please try rephrasing your question, or ask a "
    "clinician or community health worker directly."
)

GenerateFn = Callable[[List[Dict[str, str]]], str]


def build_messages(
    question: str,
    history: Optional[Sequence[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt is always first and the new user question always last.
    Invalid history turns are skipped instead of raising.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content")
            if role not in ("user", "assistant") or not content:
                continue
            messages.append({"role": role, "content": str(content)})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def needs_emergency_notice(text: str) -> bool:
    """Cheap keyword check so urgent wording is never left unflagged."""
    lowered = (text or "").lower()
    return any(hint in lowered for hint in URGENT_HINTS)


def hf_generate(
    messages: List[Dict[str, str]],
    model_id: str = MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Default generator: local Spark-X2.6-4B through transformers."""
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "transformers is not installed. Install it and download "
            f"{MODEL_ID}, or pass your own generate_fn."
        ) from exc

    generator = pipeline("text-generation", model=model_id)
    output = generator(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )

    text = output[0]["generated_text"]
    if isinstance(text, list):
        text = text[-1].get("content", "") if text else ""
    return str(text)


def answer(
    question: str,
    history: Optional[Sequence[Dict[str, str]]] = None,
    generate_fn: Optional[GenerateFn] = None,
) -> str:
    """Answer one health question with Spark-X2.6-4B.

    ``generate_fn`` receives the message list and returns text; inject a fake
    in tests so no model download or GPU is needed.
    """
    messages = build_messages(question, history)
    generator = generate_fn or hf_generate

    text = (generator(messages) or "").strip()
    if not text:
        text = FALLBACK_ANSWER

    if needs_emergency_notice(question) and EMERGENCY_LINE not in text:
        text = EMERGENCY_LINE + "\n\n" + text

    return text


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B plain-language health information assistant"
    )
    parser.add_argument("--demo", action="store_true", help="run the built-in demo question")
    parser.add_argument("--model", default=MODEL_ID, help="model id or local path")
    args = parser.parse_args(argv)

    def generate_fn(messages: List[Dict[str, str]]) -> str:
        return hf_generate(messages, model_id=args.model)

    print(DISCLAIMER)
    print()

    if args.demo:
        print("you> " + DEMO_QUESTION)
        print("spark> " + answer(DEMO_QUESTION, generate_fn=generate_fn))
        return 0

    print("Type 'exit' to quit.")
    history: List[Dict[str, str]] = []
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in ("exit", "quit"):
            return 0
        if not question:
            continue
        reply = answer(question, history=history, generate_fn=generate_fn)
        print("spark> " + reply)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    raise SystemExit(main())
