"""Minimal Spark-X2.6-4B chatbot for inclusive health knowledge.

Built for HER Hack-Astron #7 (Spark-X2.6 Open Source AI for Good Challenge).
The assistant explains health topics in plain language for people who cannot
easily reach a clinician or read formal medical text. It never diagnoses,
prescribes, or replaces professional care, and it always points the reader to
a clinician. It runs locally against ``XHToken/Spark-X2.6-4B`` via
``transformers``.

Usage::

    python src/health_chatbot.py --question "Saya demam tiga hari, apa yang bisa saya lakukan?"
    python src/health_chatbot.py            # interactive prompt
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, List, Optional, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

Message = Dict[str, str]
Generator = Callable[[List[Message]], str]

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a community health information "
    "service. You explain health topics in simple, respectful language for "
    "readers with limited medical literacy or limited literacy in general.\n"
    "Rules you must follow:\n"
    "1. You do not diagnose. Never state that the reader has a specific disease.\n"
    "2. You do not prescribe. Never give medicine names with dosages, and never "
    "tell the reader to start, stop, or change a treatment.\n"
    "3. Give general, well-established information, describe common warning "
    "signs, and explain what the reader can reasonably do next.\n"
    "4. Always direct the reader to a qualified clinician, clinic, or local "
    "emergency service for diagnosis and treatment.\n"
    "5. If the reader describes a possible emergency, say clearly that they "
    "should seek emergency care now.\n"
    "6. If you are unsure, say so instead of guessing. Do not invent sources.\n"
    "7. Keep answers short (under 200 words), use plain sentences, and avoid "
    "jargon. If the reader names a language other than English, answer in that "
    "language."
)

DEMO_QUESTION = (
    "I have had a dry cough and a mild fever for five days, about a week "
    "after returning from a trip abroad. What should I do?"
)

DISCLAIMER = (
    "This is general health information, not a diagnosis or a prescription. "
    "Please see a qualified clinician for your own situation."
)

EMERGENCY_NOTICE = (
    "Some of what you describe may need urgent care. If you have chest pain, "
    "trouble breathing, heavy bleeding, fainting, or a seizure, contact your "
    "local emergency number now."
)

EMERGENCY_SIGNALS = (
    "chest pain",
    "nyeri dada",
    "difficulty breathing",
    "trouble breathing",
    "can't breathe",
    "cannot breathe",
    "sesak napas",
    "heavy bleeding",
    "pendarahan hebat",
    "fainting",
    "pingsan",
    "seizure",
    "kejang",
    "unconscious",
    "tidak sadarkan diri",
)


def build_messages(
    question: str,
    history: Optional[Sequence[Message]] = None,
) -> List[Message]:
    """Build the chat message list sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            raise ValueError("history turns need role user/assistant and content")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def needs_emergency_notice(question: str) -> bool:
    """Heuristic check so urgent wording always gets a safety pointer."""
    text = (question or "").lower()
    return any(signal in text for signal in EMERGENCY_SIGNALS)


def format_reply(text: str, question: Optional[str] = None) -> str:
    """Turn raw model output into a reply that always carries a disclaimer."""
    body = (text or "").strip()
    if not body:
        raise ValueError("the model returned no text; nothing can be shown to the user")

    parts: List[str] = []
    if question and needs_emergency_notice(question):
        parts.append(EMERGENCY_NOTICE)
    parts.append(body)
    if DISCLAIMER not in body:
        parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def render_prompt(messages: Sequence[Message], tokenizer=None) -> str:
    """Render messages for runtimes without a chat template."""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"


def _transformers_generator(messages: List[Message]) -> str:
    """Default generator: local Spark-X2.6-4B through transformers."""
    from transformers import pipeline  # imported lazily to keep tests offline

    pipe = pipeline("text-generation", model=MODEL_ID)
    outputs = pipe(
        list(messages),
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    generated = outputs[0]["generated_text"]
    if isinstance(generated, list):
        return str(generated[-1].get("content", ""))
    return str(generated)


class HealthChatBot:
    """Thin wrapper around Spark-X2.6-4B with guardrails and lazy loading."""

    def __init__(
        self,
        generator: Optional[Generator] = None,
        model_id: str = MODEL_ID,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._generator = generator

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = _transformers_generator
        return self._generator

    def answer(
        self,
        question: str,
        history: Optional[Sequence[Message]] = None,
    ) -> str:
        messages = build_messages(question, history)
        raw = self.generator(messages)
        return format_reply(raw, question)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health information chatbot")
    parser.add_argument("--question", "-q", default=None, help="question to ask once")
    parser.add_argument("--model", default=MODEL_ID, help="model id or local path")
    args = parser.parse_args(argv)

    bot = HealthChatBot(model_id=args.model)

    if args.question:
        print(bot.answer(args.question))
        return 0

    print(f"Spark-X2.6 health information demo (model: {args.model}). Ctrl-D to exit.")
    print("General information only, not medical advice.\n")
    history: List[Message] = []
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        reply = bot.answer(question, history)
        print(f"spark> {reply}\n")
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reply},
            ]
        )


if __name__ == "__main__":
    sys.exit(main())
