"""Minimal health-information chatbot built on Spark-X2.6-4B (AI for Good demo).

This is an educational prototype: it explains health information in plain
language and always points the user to a qualified clinician. It does not
diagnose, prescribe, or replace professional care.

Run the demo prompt without downloading weights::

    python src/health_chatbot.py --show-prompt

Run it against a local copy of the model (requires ``transformers``)::

    python src/health_chatbot.py --demo
    python src/health_chatbot.py "Apa yang perlu saya perhatikan jika demam 3 hari?"
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in plain, "
    "respectful language that a non-specialist can understand. "
    "You must NOT diagnose, prescribe, or recommend specific medicines or doses. "
    "Never claim to replace a clinician. Always end with a short reminder that the "
    "person should consult a qualified healthcare professional, and mention urgent "
    "care signs when the described symptoms could be serious. "
    "If information is missing, say what is missing instead of guessing."
)

DEMO_QUESTION = (
    "Saya baru pulang dari perjalanan dan sekarang batuk kering plus demam 38 derajat "
    "Celsius sejak dua hari. Apa yang sebaiknya saya perhatikan sebelum ke klinik?"
)

SAFETY_NOTE = (
    "This is general health education, not medical advice. "
    "For urgent symptoms, contact local emergency services."
)


def build_messages(
    question: str, history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B.

    ``history`` is a list of prior ``{"role": ..., "content": ...}`` turns and is
    copied, never mutated.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("history entries must be dicts with role and content")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant"):
            raise ValueError("history role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("history content must be a non-empty string")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def render_prompt(messages: List[Dict[str, str]]) -> str:
    """Fallback ChatML rendering for runtimes without a chat template."""
    parts = [
        "<|im_start|>{role}\n{content}<|im_end|>".format(
            role=message["role"], content=message["content"]
        )
        for message in messages
    ]
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def extract_reply(outputs) -> str:
    """Extract the assistant text from a transformers text-generation result."""
    if not outputs:
        raise ValueError("model returned no output")

    first = outputs[0]
    if isinstance(first, dict):
        generated = first.get("generated_text", first.get("text", ""))
    else:
        generated = first

    if isinstance(generated, (list, tuple)):
        if not generated:
            raise ValueError("model returned an empty conversation")
        last = generated[-1]
        text = last.get("content", "") if isinstance(last, dict) else str(last)
    else:
        text = str(generated)

    text = text.strip()
    if not text:
        raise ValueError("model returned an empty reply")
    return text


def with_safety_note(reply: str) -> str:
    """Append the fixed safety note once, without duplicating it."""
    text = (reply or "").strip()
    if SAFETY_NOTE in text:
        return text
    return text + "\n\n" + SAFETY_NOTE


def _load_pipeline():
    """Load Spark-X2.6-4B lazily so importing this module stays dependency-free."""
    from transformers import pipeline

    return pipeline("text-generation", model=MODEL_ID, device_map="auto")


def generate_reply(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    generator=None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Ask Spark-X2.6-4B for a plain-language explanation.

    ``generator`` can be injected (for tests or a different runtime) and must be
    callable like the transformers ``text-generation`` pipeline.
    """
    messages = build_messages(question, history)
    if generator is None:
        generator = _load_pipeline()
    outputs = generator(messages, max_new_tokens=max_new_tokens, do_sample=False)
    return extract_reply(outputs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot (educational demo)"
    )
    parser.add_argument("question", nargs="?", default=None, help="health question")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo question")
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the assembled prompt only (no model download)",
    )
    args = parser.parse_args(argv)

    question = DEMO_QUESTION if args.demo or not args.question else args.question
    messages = build_messages(question)

    if args.show_prompt:
        print(render_prompt(messages))
        return 0

    try:
        reply = generate_reply(question, generator=None)
    except ImportError:
        print(
            "transformers is required to run the model: "
            "pip install transformers accelerate",
            file=sys.stderr,
        )
        return 2
    except ValueError as error:
        print("Could not read a reply from the model: %s" % error, file=sys.stderr)
        return 1

    print("Q: " + question)
    print()
    print("A: " + reply)
    print()
    print(with_safety_note(reply).split(SAFETY_NOTE)[-1].strip() or SAFETY_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
