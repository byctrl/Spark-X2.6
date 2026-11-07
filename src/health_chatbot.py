"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Runs fully locally against ``XHToken/Spark-X2.6-4B``. The assistant is scoped to
plain-language patient education: it does not diagnose, prescribe, or replace a
clinician. If the model is unavailable the demo fails loudly instead of
inventing an answer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in simple, "
    "respectful language for a general audience. You do not diagnose, you do not "
    "prescribe or recommend prescription medicines or doses, and you do not tell "
    "the user to stop or change a treatment. State clearly when the question needs "
    "a qualified clinician, and suggest what to tell that clinician. Mention urgent "
    "care when the described symptoms could be an emergency. Answer only from "
    "general knowledge; say so when you are unsure."
)

DISCLAIMER = (
    "This is general health information, not medical advice. For diagnosis or "
    "treatment, consult a qualified clinician."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have had a dry cough "
    "and a mild fever in the evenings. What could this be, and when should I see "
    "a doctor?"
)

_VALID_ROLES = ("user", "assistant")


def build_messages(
    question: str,
    history: Optional[Sequence[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat messages for one turn, keeping the safety system prompt."""
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in _VALID_ROLES and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": cleaned})
    return messages


def build_prompt(
    messages: Sequence[Dict[str, str]],
    tokenizer: Optional[object] = None,
) -> str:
    """Render messages with the model chat template, or a plain fallback."""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
    lines = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
    lines.append("<|assistant|>")
    return "\n".join(lines)


def clean_answer(text: object) -> str:
    """Normalise raw model output into a non-empty trimmed string."""
    if isinstance(text, (list, tuple)):
        text = text[-1] if text else ""
    if not isinstance(text, str):
        return ""
    return text.strip()


def format_reply(answer: str) -> str:
    """Append the fixed safety disclaimer to a model answer."""
    body = clean_answer(answer)
    if not body:
        return DISCLAIMER
    return f"{body}\n\n{DISCLAIMER}"


@dataclass
class HealthChatbot:
    """Thin local wrapper around a text-generation pipeline for Spark-X2.6-4B."""

    model_id: str = MODEL_ID
    max_new_tokens: int = MAX_NEW_TOKENS
    _pipeline: object = field(default=None, init=False, repr=False)

    def load(self):
        """Lazily create the pipeline so tests do not need model weights."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "transformers is required to run the demo: "
                    "pip install transformers torch"
                ) from exc
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                device_map="auto",
            )
        return self._pipeline

    def answer(
        self,
        question: str,
        history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> str:
        messages = build_messages(question, history)
        pipe = self.load()
        prompt = build_prompt(messages, getattr(pipe, "tokenizer", None))
        outputs = pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        if not outputs:
            return ""
        first = outputs[0]
        raw = first.get("generated_text", "") if isinstance(first, dict) else first
        return clean_answer(raw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health-information demo")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args(argv)

    bot = HealthChatbot(model_id=args.model)
    print(f"model: {args.model}")
    print(f"question: {args.question}\n")
    print(format_reply(bot.answer(args.question)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
