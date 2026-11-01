"""Minimal accessible-health-information chatbot demo for Spark-X2.6-4B.

This is a small public-interest prototype: it helps a person restate a
health question in plain language and points them to professional care.
It does NOT diagnose, prescribe, or replace a clinician.

The model is only loaded when ``generate_reply`` / the CLI is actually
invoked, so importing this module (and running the tests) needs no GPU,
no network access, and no heavy dependencies.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant in a public-interest health "
    "information tool. Explain health topics in simple, plain language at "
    "roughly a 6th-grade reading level. Use short paragraphs and bullet "
    "points where helpful.\n\n"
    "Strict rules:\n"
    "1. Never give a diagnosis, a prescription, a dosage, or a treatment "
    "plan.\n"
    "2. Never tell the person to stop, start, or change a medication.\n"
    "3. Encourage the person to contact a qualified clinician, pharmacist, "
    "or local health service for anything specific to their situation.\n"
    "4. Describe warning signs that mean the person should seek urgent "
    "care, and say clearly that this is not a substitute for medical "
    "advice.\n"
    "5. If the question falls outside general health information, say so "
    "briefly and redirect to an appropriate professional.\n"
    "6. Do not ask for or repeat identifying details such as full name, "
    "address, ID numbers, or insurance numbers."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have had a dry "
    "cough and a mild fever in the evenings. I do not have a doctor yet. "
    "What should I know before I find one?"
)


def build_messages(question: str, extra_context: Optional[str] = None) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B.

    Raises ValueError when the question is empty or not a string, so a
    caller can show a friendly prompt instead of querying the model.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    user_content = question.strip()
    if extra_context and extra_context.strip():
        user_content = user_content + "\n\nExtra context:\n" + extra_context.strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_prompt(messages: List[Dict[str, str]]) -> str:
    """Render messages with a simple role-tag template.

    Used as a fallback when the tokenizer has no chat template.
    """
    parts = []
    for message in messages:
        parts.append("<|{0}|>\n{1}".format(message["role"], message["content"]))
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def generate_reply(
    question: str,
    extra_context: Optional[str] = None,
    model_id: str = MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate a plain-language answer with Spark-X2.6-4B.

    ``transformers`` is imported here so the rest of the module (and the
    unit tests) work without the model stack installed.
    """
    from transformers import pipeline  # lazy import: keeps tests lightweight

    messages = build_messages(question, extra_context)
    prompt = build_prompt(messages)

    generator = pipeline("text-generation", model=model_id)
    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return outputs[0]["generated_text"].strip()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a general health-information question "
        "(education only, not medical advice)."
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--context", default=None, help="optional extra context")
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args(argv)

    try:
        answer = generate_reply(args.question, args.context, model_id=args.model)
    except ValueError as exc:
        print("Input error: {0}".format(exc), file=sys.stderr)
        return 2
    except ImportError:
        print(
            "The model stack is not installed. Run: pip install transformers torch",
            file=sys.stderr,
        )
        return 3

    print(answer)
    print("\n---\nThis is general health information, not medical advice.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
