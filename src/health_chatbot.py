"""Minimal Spark-X2.6-4B chatbot for inclusive health knowledge access.

Public-interest demo for HER Hack-Astron #7: a plain-language patient
education assistant. The model only explains general health topics; it must
never diagnose, prescribe, or replace a clinician.

Usage:
    python src/health_chatbot.py --demo
    python src/health_chatbot.py --question "What are common causes of a dry cough?"

The model is loaded lazily, so importing this module (and running the tests)
needs no GPU, no network access and no third-party packages.
"""

from __future__ import annotations

import argparse
import json
from typing import Callable, Iterable, List, Optional, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

# Role prompt: patient education only, simple language, no diagnosis.
SYSTEM_PROMPT = (
    "You are a patient-education assistant for a free public health "
    "information service. Explain general health topics in plain, simple "
    "language that someone without medical training can understand. "
    "Do not diagnose the reader, do not name prescription medicines or "
    "dosages, and do not claim to replace a clinician. Always remind the "
    "reader to contact a qualified health professional for diagnosis and "
    "treatment, and to seek emergency care immediately for severe or "
    "worsening symptoms (for example difficulty breathing or chest pain). "
    "If the question is unclear, ask one short clarifying question. "
    "Never ask for or repeat personal identifying information."
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "Please talk to a qualified clinician about your own situation."
)

# Demo scenario: dry cough and mild fever two days after travelling.
DEMO_QUESTION = (
    "Saya baru pulang dari perjalanan dan sekarang batuk kering serta demam "
    "ringan sejak dua hari lalu. Apa yang sebaiknya saya perhatikan?"
)

Generator = Callable[[Sequence[dict]], str]


def build_messages(
    question: str,
    history: Optional[Iterable[dict]] = None,
) -> List[dict]:
    """Build the chat messages: system prompt, optional history, question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def render_prompt(messages: Sequence[dict], tokenizer=None) -> str:
    """Render messages to a text prompt, using the chat template if present."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
    return "\n\n".join(
        "{0}: {1}".format(message["role"].upper(), message["content"])
        for message in messages
    )


class SparkGenerator:
    """Thin lazy wrapper around the Spark-X2.6-4B transformers pipeline."""

    def __init__(self, model_id: str = MODEL_ID, max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    def __call__(self, messages: Sequence[dict]) -> str:
        prompt = render_prompt(messages, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_reply(
    question: str,
    history: Optional[Iterable[dict]] = None,
    generator: Optional[Generator] = None,
    model_id: str = MODEL_ID,
) -> str:
    """Return the model's plain-language answer for one question."""
    messages = build_messages(question, history)
    if generator is None:
        generator = SparkGenerator(model_id=model_id)
    reply = (generator(messages) or "").strip()
    if not reply:
        raise RuntimeError("model returned an empty reply")
    return reply


def format_answer(reply: str) -> str:
    """Attach the non-medical-advice disclaimer to a model reply."""
    return "{0}\n\n{1}".format((reply or "").strip(), DISCLAIMER)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health education chatbot")
    parser.add_argument("--question", help="question to ask the model")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo question")
    parser.add_argument("--model", default=MODEL_ID, help="model id or local path")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    args = parser.parse_args(argv)

    question = args.question or DEMO_QUESTION
    reply = generate_reply(question, model_id=args.model)
    if args.json:
        print(json.dumps(
            {"model": args.model, "question": question, "answer": reply, "disclaimer": DISCLAIMER},
            ensure_ascii=False,
            indent=2,
        ))
    else:
        print(format_answer(reply))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
