"""Minimal health-knowledge chatbot built on Spark-X2.6-4B.

Purpose (AI for Good / health + accessibility):
    Turn a plain-language health question into a short educational answer that
    is easy to read and always points the user to a real clinician.

Safety boundary:
    This is *not* a diagnostic tool. The system prompt forbids diagnosis,
    prescriptions and dose changes, and requires a clinician referral. The
    application never treats model output as medical ground truth.

Run the demo:
    python src/health_chatbot.py            # uses DEMO_QUESTION
    python src/health_chatbot.py "my question here" --json

The heavy dependencies (torch/transformers) are imported lazily so the module
and its tests stay importable on a CPU-only machine without the model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Optional

MODEL_ID = os.environ.get("SPARK_MODEL_ID", "XHToken/Spark-X2.6-4B")
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public health information service.\n"
    "Follow these rules in every answer:\n"
    "1. Use simple, everyday language and short sentences. Avoid medical jargon; "
    "if a technical term is unavoidable, explain it in one short line.\n"
    "2. Do NOT diagnose, do NOT name a definitive disease, and do NOT prescribe or "
    "change any medicine or dose.\n"
    "3. Explain in general terms what the described signs may mean, what the person "
    "can safely do in the meantime, and which warning signs mean 'seek care now'.\n"
    "4. Always end by telling the user to confirm with a qualified clinician or the "
    "local health service.\n"
    "5. If the message suggests an emergency (for example chest pain, difficulty "
    "breathing, heavy bleeding, fainting), say clearly and first: contact emergency "
    "services now.\n"
    "6. Answer in the language of the user's question.\n"
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak tiga hari lalu, setelah pulang dari "
    "perjalanan. Apa yang sebaiknya saya perhatikan?"
)

REFERRAL_MARKERS = (
    "clinician",
    "doctor",
    "physician",
    "nurse",
    "health service",
    "health centre",
    "health center",
    "clinic",
    "dokter",
    "tenaga kesehatan",
    "puskesmas",
    "klinik",
)


def build_messages(
    question: str,
    history: Optional[Iterable[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt always comes first and can never be replaced by history.
    Only non-empty user/assistant turns from ``history`` are kept.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = (turn or {}).get("role")
        content = ((turn or {}).get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def format_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Render messages with the model chat template, with a plain fallback."""
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    lines: List[str] = []
    for message in messages:
        lines.append("<|{0}|>\n{1}".format(message["role"], message["content"]))
    lines.append("<|assistant|>\n")
    return "\n".join(lines)


def load_pipeline(model_id: str = MODEL_ID, device_map: Optional[str] = None):
    """Load a text-generation pipeline for Spark-X2.6-4B (lazy import)."""
    from transformers import pipeline

    kwargs = {}
    if device_map:
        kwargs["device_map"] = device_map
    return pipeline("text-generation", model=model_id, **kwargs)


def generate(
    question: str,
    history: Optional[Iterable[Dict[str, str]]] = None,
    model_id: str = MODEL_ID,
    pipe=None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Answer ``question`` with Spark-X2.6-4B and return the trimmed text."""
    messages = build_messages(question, history)
    if pipe is None:
        pipe = load_pipeline(model_id)
    prompt = format_prompt(pipe.tokenizer, messages)
    output = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return (output[0]["generated_text"] or "").strip()


def has_clinician_referral(answer: str) -> bool:
    """Cheap post-check that the safety instruction was actually followed."""
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in REFERRAL_MARKERS)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a health question (education only)."
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    answer = generate(
        args.question,
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "question": args.question,
                    "answer": answer,
                    "has_clinician_referral": has_clinician_referral(answer),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(answer)

    if not has_clinician_referral(answer):
        print(
            "[notice] The answer did not mention a clinician referral. "
            "For educational use only; not a diagnosis.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
