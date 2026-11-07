"""Minimal health-information chatbot for HER Hack-Astron #7.

Purpose
-------
A tiny, dependency-free-on-import demo that actually uses
``XHToken/Spark-X2.6-4B`` to answer patient-education questions in plain
language. The model never diagnoses or prescribes; it points the user to a
qualified clinician.

Run locally (needs ``pip install transformers torch`` and the weights):

    python src/health_chatbot.py
    python src/health_chatbot.py "Saya demam tiga hari, apa yang perlu saya waspadai?"

Tests run without the model, using stdlib ``unittest``:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information chatbot. Explain health topics in plain language at a "
    "sixth-grade reading level, in the same language as the user. "
    "You must not diagnose, prescribe, or recommend a specific medicine or "
    "dose, and you must not give a personal treatment plan. Instead, list "
    "general warning signs that mean the person should seek urgent care, "
    "and encourage them to consult a qualified clinician. State clearly "
    "when information is uncertain, and never invent sources, statistics, "
    "or hotline numbers."
)

DEMO_QUESTION = (
    "Saya batuk dan demam sejak tiga hari lalu setelah pulang dari "
    "perjalanan. Apa yang sebaiknya saya perhatikan?"
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "Please see a qualified clinician for your own situation."
)

Message = Dict[str, str]
Generator = Callable[[List[Message]], str]


def build_messages(question: str, system_prompt: str = SYSTEM_PROMPT) -> List[Message]:
    """Build the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": question.strip()},
    ]


def format_with_disclaimer(reply: str) -> str:
    """Always ship a safety disclaimer with the model output."""
    text = (reply or "").strip()
    if not text:
        return DISCLAIMER
    if DISCLAIMER in text:
        return text
    return text + "\n\n" + DISCLAIMER


def load_transformers_generator(model_id: str = MODEL_ID) -> Generator:
    """Load Spark-X2.6-4B and return a ``messages -> text`` callable."""
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "transformers is required for real inference: "
            "pip install transformers torch"
        ) from exc

    pipe = pipeline("text-generation", model=model_id, device_map="auto")

    def generate(messages: List[Message]) -> str:
        outputs = pipe(
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_full_text=False,
        )
        generated = outputs[0]["generated_text"]
        if isinstance(generated, list):
            last = generated[-1] if generated else {}
            if isinstance(last, dict):
                return str(last.get("content", ""))
            return str(last)
        return str(generated)

    return generate


def answer_question(question: str, generate: Optional[Generator] = None) -> str:
    """Answer one question with Spark-X2.6-4B plus the safety disclaimer."""
    messages = build_messages(question)
    generator = generate if generate is not None else load_transformers_generator()
    return format_with_disclaimer(generator(messages))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B plain-language health information demo"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEMO_QUESTION,
        help="question to ask (defaults to the demo scenario)",
    )
    args = parser.parse_args(argv)
    print(answer_question(args.question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
