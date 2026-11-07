"""Minimal AI-for-Good demo: a health-information chatbot on Spark-X2.6-4B.

Run:
    python src/health_chatbot.py --demo
    python src/health_chatbot.py --question "..."

The model is loaded lazily, so importing this module has no third-party
dependency and the prompt/guard logic can be tested without a GPU.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, List, Optional, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant helping people understand general "
    "health information in plain language.\n"
    "Rules:\n"
    "1. Answer in the language the user writes in, using short sentences and "
    "everyday words.\n"
    "2. Never diagnose, never state that the user has a named condition, and "
    "never prescribe, give doses, or tell the user to change a treatment.\n"
    "3. Explain in general terms what such symptoms may relate to, what a "
    "clinician will typically check, and which warning signs need urgent care.\n"
    "4. If the question is unclear or outside general health education, say so "
    "and suggest who to ask instead.\n"
    "5. End with a short reminder that this is education, not medical advice."
)

DISCLAIMER = (
    "This is general health education, not medical advice. It cannot diagnose "
    "or prescribe. Please consult a qualified clinician, and seek emergency "
    "care right away for severe or quickly worsening symptoms."
)

DEMO_QUESTION = (
    "I have had a dry cough and a mild fever for three days since I came back "
    "from a trip. What should I know before I see a clinician?"
)


def build_messages(question: str) -> List[Dict[str, str]]:
    """Return the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def with_disclaimer(answer: Optional[str]) -> str:
    """Attach the safety notice exactly once; never return an empty answer."""
    text = (answer or "").strip()
    if not text:
        return DISCLAIMER
    if DISCLAIMER in text:
        return text
    return "{}\n\n{}".format(text, DISCLAIMER)


def _default_generator(messages, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Load Spark-X2.6-4B with transformers on first use."""
    from transformers import pipeline  # imported lazily on purpose

    pipe = pipeline("text-generation", model=MODEL_ID)
    result = pipe(list(messages), max_new_tokens=max_new_tokens, do_sample=False)
    generated = result[0]["generated_text"]
    if isinstance(generated, list):
        return str(generated[-1].get("content", ""))
    return str(generated)


def generate_reply(
    question: str,
    generator: Optional[Callable[..., str]] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Ask Spark-X2.6-4B one health question and return a guarded answer."""
    messages = build_messages(question)
    gen = generator if generator is not None else _default_generator
    raw = gen(messages, max_new_tokens=max_new_tokens)
    return with_disclaimer(raw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot (AI for Good demo)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--question", help="health question to ask")
    group.add_argument("--demo", action="store_true", help="run the built-in demo question")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    question = args.question or (DEMO_QUESTION if args.demo else None)
    if not question:
        parser.print_help()
        return 2

    print(generate_reply(question, max_new_tokens=args.max_new_tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
