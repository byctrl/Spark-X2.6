"""Minimal health-information chatbot built on Spark-X2.6-4B.

Scope: patient education only. The model is asked for plain-language general
information, never for diagnosis or treatment, and every reply ends with a
reminder to confirm with a qualified clinician.

Run locally (needs a machine that can host the model):

    python src/health_chatbot.py
    python src/health_chatbot.py --question "What is a fever?"
"""

from __future__ import annotations

import argparse
import sys

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people with limited access to "
    "health information. Explain things in plain, simple language. "
    "Do not diagnose, do not prescribe, and do not give medicine names, doses "
    "or treatment plans. If the situation sounds urgent, tell the reader to "
    "contact local emergency services right away. Never claim to replace a "
    "clinician, and end by recommending that the reader confirm with a "
    "qualified health worker."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago. Now I have a cough and a mild "
    "fever in the evening. I do not have a doctor nearby. What should I know?"
)

DISCLAIMER = (
    "This is general education, not medical advice. "
    "Please confirm with a qualified health worker."
)

URGENT_BANNER = (
    "If this may be an emergency, contact your local emergency number now."
)

EMERGENCY_HINTS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "unconscious",
    "not breathing",
    "severe bleeding",
    "suicide",
    "overdose",
)


def build_messages(question: str) -> list[dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def looks_urgent(text: str) -> bool:
    """Cheap keyword screen so the app can prepend an emergency reminder."""
    lowered = (text or "").lower()
    return any(hint in lowered for hint in EMERGENCY_HINTS)


def format_reply(answer: str, urgent: bool = False) -> str:
    """Wrap the model answer with the safety banner and the disclaimer."""
    answer = (answer or "").strip()
    if not answer:
        answer = "No answer was produced. Please try rephrasing your question."
    parts = []
    if urgent:
        parts.append(URGENT_BANNER)
    parts.append(answer)
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def extract_text(output) -> str:
    """Pull the assistant text out of a transformers text-generation result."""
    if not output:
        return ""
    item = output[0]
    generated = item.get("generated_text", "") if isinstance(item, dict) else item
    if isinstance(generated, list):
        if not generated:
            return ""
        last = generated[-1]
        if isinstance(last, dict):
            return str(last.get("content", "") or "")
        return str(last)
    return str(generated or "")


def load_pipeline():
    """Load Spark-X2.6-4B. Imported lazily so tests need no torch install."""
    from transformers import pipeline  # noqa: PLC0415

    return pipeline("text-generation", model=MODEL_ID, device_map="auto")


def generate_reply(question: str, generator=None) -> str:
    """Ask Spark-X2.6-4B for a plain-language answer and apply safety wrapping."""
    messages = build_messages(question)
    if generator is None:
        generator = load_pipeline()
    output = generator(messages, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return format_reply(extract_text(output), urgent=looks_urgent(question))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot (education only)"
    )
    parser.add_argument("--question", default=None, help="question to ask")
    args = parser.parse_args(argv)

    question = args.question
    if question is None:
        question = input("Your health question: ").strip() or DEMO_QUESTION

    print(generate_reply(question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
