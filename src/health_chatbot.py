#!/usr/bin/env python3
"""Minimal health-information chatbot built on Spark-X2.6-4B.

AI for Good entry for HER Hack-Astron #7 (XHToken/Spark-X2.6-4B).

Scope and safety
----------------
The assistant explains general health topics in plain language so that people
with limited access to health information can prepare better questions for a
clinician. It is explicitly *not* a diagnostic or prescribing tool:

* the system prompt forbids diagnosis, dosing and treatment plans;
* every reply is post-processed to carry a fixed disclaimer;
* a trained clinician remains the decision maker.

Spark-X2.6-4B is a text-only model: no image input and no OCR. Text that comes
from a photo or a scanned notice must be extracted by a separate step and passed
in here as plain text.

Run the demo (needs `transformers` and enough memory for the 4B weights)::

    python src/health_chatbot.py
    python src/health_chatbot.py "How can I prepare for a doctor visit?"

The helpers (`build_messages`, `build_prompt`, `generate_reply`) do not import a
runtime, so the CLI wiring can be swapped for a local OpenAI-compatible server
without touching the safety layer.
"""

from __future__ import annotations

import argparse
from typing import Callable, Iterable, Mapping, Optional, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

DEMO_QUESTION = (
    "I came back from a two-week trip and since then I have a cough and a mild "
    "fever. What should I pay attention to and what should I ask a doctor?"
)

DISCLAIMER = (
    "This is general health information for education only, not a diagnosis, "
    "prescription or treatment plan. Please discuss any symptom, medicine or "
    "test result with a qualified clinician, and contact local emergency "
    "services if the situation may be urgent."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who may have limited "
    "access to health information. Answer in the language of the user's "
    "question, using short sentences and everyday words. Explain general "
    "background, the questions it is worth asking a clinician, and warning "
    "signs that need urgent care.\n"
    "Hard rules:\n"
    "- Never give a diagnosis, a diagnosis-like probability, a prescription, a "
    "dose, or a treatment plan.\n"
    "- Never tell the user to start, stop or change prescribed medication.\n"
    "- If the question asks for a diagnosis or a dose, say clearly that you "
    "cannot provide it, and explain what a clinician needs in order to decide.\n"
    "- Point the user to a qualified clinician or to local emergency services "
    "when the described situation could be urgent.\n"
    "- Do not ask for or repeat identifying details such as names, addresses, "
    "ID numbers or insurance numbers.\n"
    "- Say clearly when you do not know something instead of guessing."
)

ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}
HISTORY_ROLES = ("user", "assistant")


def build_messages(
    question: str,
    history: Optional[Sequence[Mapping[str, str]]] = None,
) -> list:
    """Return the chat messages for one turn: system prompt + history + question.

    ``history`` is an optional sequence of ``{"role": ..., "content": ...}``
    mappings. Only ``user`` and ``assistant`` turns are kept; other roles are
    ignored so a caller cannot override the system prompt.
    """
    text = str(question).strip() if question is not None else ""
    if not text:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if role in HISTORY_ROLES and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})
    return messages


def build_prompt(
    question: str,
    history: Optional[Sequence[Mapping[str, str]]] = None,
) -> str:
    """Flatten the chat messages into a plain prompt for a base text model."""
    parts = [
        "%s: %s" % (ROLE_LABELS.get(message["role"], message["role"]), message["content"])
        for message in build_messages(question, history)
    ]
    parts.append("Assistant:")
    return "\n\n".join(parts)


def with_disclaimer(answer: str) -> str:
    """Append the fixed disclaimer once, even if the model already wrote it."""
    text = (answer or "").strip()
    if not text:
        text = "(no answer was generated)"
    if DISCLAIMER in text:
        return text
    return "%s\n\n%s" % (text, DISCLAIMER)


def generate_reply(
    question: str,
    generate: Callable[[str, int], str],
    history: Optional[Sequence[Mapping[str, str]]] = None,
) -> str:
    """Generate one safety-wrapped reply.

    ``generate`` is any callable taking ``(prompt, max_new_tokens)`` and
    returning raw model text. Keeping it injectable makes the safety layer
    testable without downloading model weights.
    """
    prompt = build_prompt(question, history)
    raw = generate(prompt, MAX_NEW_TOKENS)
    return with_disclaimer(raw)


def load_generator(model_id: str = MODEL_ID) -> Callable[[str, int], str]:
    """Load Spark-X2.6-4B lazily and return a ``(prompt, max_new_tokens)`` callable."""
    from transformers import pipeline  # imported lazily: keeps tests dependency-free

    pipe = pipeline("text-generation", model=model_id, device_map="auto")

    def generate(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        outputs = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return outputs[0]["generated_text"]

    return generate


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health-information demo")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--model", default=MODEL_ID)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    generate = load_generator(args.model)
    print(generate_reply(args.question, generate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
