"""Minimal health-information chatbot demo powered by Spark-X2.6-4B.

Public-interest (health / accessibility) entry for HER Hack-Astron #7.
The assistant explains general health information in plain language.
It does NOT diagnose, prescribe, or replace a clinician, and every answer is
paired with the disclaimer below.

Spark-X2.6-4B is a text-only model: any OCR or image handling would have to be
added by the client before calling this module.

Run the built-in demo:
    python src/health_chatbot.py --demo
Ask your own question:
    python src/health_chatbot.py --question "..."
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who may be far from a clinic "
    "or reading in a second language. Explain general health information in plain, "
    "respectful language a non-specialist can understand, using short paragraphs or "
    "a short list. "
    "Never diagnose, never prescribe, never give a drug dose, and never tell the user "
    "to start, stop, or change a treatment. Do not claim to be a clinician or to have "
    "examined the user. "
    "Always encourage the user to consult a qualified health professional, and clearly "
    "list the warning signs that mean the person should seek urgent in-person care. "
    "If the question is outside general health education, say so briefly and point to a "
    "professional or a trusted local service instead of guessing. "
    "Do not ask for names, addresses, ID numbers, or other identifying details. "
    "Answer in the language the user writes in."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak kembali dari perjalanan lima hari "
    "yang lalu. Apa yang sebaiknya saya perhatikan sebelum memeriksakan diri ke klinik?"
)

DISCLAIMER = (
    "This is general health education, not medical advice and not a diagnosis. "
    "Please consult a qualified health professional about your own situation."
)


def build_messages(question: str, history: Optional[Iterable[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    """Build the chat messages sent to the model.

    The system prompt always comes first and the current question always comes
    last, so callers cannot accidentally override the safety instructions.
    History turns with unknown roles or empty content are dropped.
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": (question or "").strip()})
    return messages


def build_prompt(question: str, history: Optional[Iterable[Dict[str, str]]] = None) -> str:
    """Plain-text fallback for runtimes without a chat template."""
    parts = []
    for message in build_messages(question, history):
        parts.append("<|{0}|>\n{1}".format(message["role"], message["content"]))
    parts.append("<|assistant|>")
    return "\n".join(parts)


def generate(
    question: str,
    history: Optional[Iterable[Dict[str, str]]] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    model_id: str = MODEL_ID,
) -> str:
    """Generate one answer with Spark-X2.6-4B.

    transformers/torch are imported here so that the prompt helpers above (and
    their tests) work without the weights or the ML stack installed.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    messages = build_messages(question, history)
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = build_prompt(question, history)

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    completion = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot (education only, not medical advice)."
    )
    parser.add_argument("--question", "-q", default="", help="health question to ask")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo question")
    parser.add_argument("--model", default=MODEL_ID, help="model id or local path")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    question = DEMO_QUESTION if (args.demo or not args.question.strip()) else args.question

    print("Question: " + question)
    try:
        answer = generate(
            question,
            max_new_tokens=args.max_new_tokens,
            model_id=args.model,
        )
    except Exception as exc:  # missing ML stack, missing weights, no device, ...
        print("Could not run {0}: {1}".format(args.model, exc), file=sys.stderr)
        print(
            "Install torch/transformers and download the model, or pass --model with a local path.",
            file=sys.stderr,
        )
        return 2

    print("\nAnswer:\n" + answer)
    print("\n" + DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
