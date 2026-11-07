"""Minimal Spark-X2.6-4B health-information chatbot.

HER Hack-Astron #7 entry (AI for Good: inclusive access to health knowledge).

The model answers general health-education questions in plain language and is
explicitly barred from diagnosing or prescribing. Every answer points the user
to a qualified clinician, and the script short-circuits to an emergency message
when the question contains a red flag, so the model is never asked to triage.

Run:
    python src/health_chatbot.py --question "..."

The model is loaded lazily, so importing this module (and running the tests)
works without transformers, torch or the 4B weights installed.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health topics in plain "
    "language for people with limited medical literacy.\n"
    "Rules:\n"
    "1. Never diagnose and never prescribe or change medication.\n"
    "2. If the question suggests an emergency, tell the user to contact "
    "emergency services or a clinician right now.\n"
    "3. Always recommend confirming with a qualified clinician.\n"
    "4. Say clearly when you are unsure instead of guessing.\n"
    "5. Keep answers short (under 200 words), use short sentences, and define "
    "any medical term you use."
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "Please confirm with a qualified clinician."
)

FALLBACK_REPLY = (
    "I could not produce an explanation just now. Please try rephrasing your "
    "question, or ask a qualified clinician."
)

EMERGENCY_REPLY = (
    "This may be an emergency. Please contact your local emergency number or "
    "go to the nearest emergency department now. I am an education tool only "
    "and cannot assess or treat you."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and now I have a cough and a mild "
    "fever. What could it be and should I take antibiotics?"
)

RED_FLAGS = (
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "shortness of breath",
    "severe bleeding",
    "unconscious",
    "suicidal",
    "seizure",
    "stiff neck",
)


def has_red_flag(question: str) -> bool:
    """Return True when the question mentions a possible emergency."""
    text = (question or "").lower()
    return any(flag in text for flag in RED_FLAGS)


def build_messages(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B."""
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("question must not be empty")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    for turn in history or []:
        role = str(turn.get("role", "")).strip()
        content = str(turn.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            raise ValueError("history turns need role user/assistant and content")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": cleaned})
    return messages


def answer(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    generator: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> str:
    """Return the chatbot reply for one question.

    ``generator`` is any callable taking chat messages and returning text; it
    defaults to a lazily loaded Spark-X2.6-4B pipeline.
    """
    messages = build_messages(question, history)
    if has_red_flag(question):
        return EMERGENCY_REPLY

    run = generator if generator is not None else SparkGenerator()
    text = (run(messages) or "").strip()
    return text or FALLBACK_REPLY


class SparkGenerator:
    """Thin wrapper around a local Spark-X2.6-4B text-generation pipeline."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            # Imported lazily so the tests never need torch/transformers.
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-generation",
                model=self.model_id,
            )
        return self._pipeline

    def _render(self, messages: List[Dict[str, str]]) -> str:
        pipe = self._load()
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        lines = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        return "\n\n".join(lines) + "\n\nASSISTANT:"

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        prompt = self._render(messages)
        result = self._load()(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return result[0]["generated_text"]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a plain-language health question."
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    generator = SparkGenerator(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )
    print(answer(args.question, generator=generator))
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
