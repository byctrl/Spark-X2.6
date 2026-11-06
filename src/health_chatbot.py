"""Minimal Spark-X2.6-4B health-information chatbot.

A small, dependency-light demo for the HER Hack-Astron #7 AI-for-Good
challenge: inclusive, plain-language health education. Spark-X2.6-4B is
used only to explain general health information. It must not diagnose,
dose, prescribe, or replace a clinician.
"""

from __future__ import annotations

import argparse
import sys

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain general health "
    "information in plain language for readers with low medical literacy, "
    "avoiding unexplained jargon and keeping sentences short.\n"
    "Rules you must always follow:\n"
    "1. Do not diagnose, do not present a likely disease as a conclusion, "
    "and do not give a prescription, dose, or treatment plan.\n"
    "2. Do not claim to replace a doctor, nurse, or pharmacist.\n"
    "3. Say clearly when a symptom needs urgent in-person care, and advise "
    "contacting a local clinician or emergency service.\n"
    "4. If information is missing or uncertain, say so instead of guessing.\n"
    "5. Answer in the language the user writes in."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days, and I got back from "
    "a long trip last week. What should I know, and when should I see a "
    "doctor?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    ``history`` is an optional list of earlier turns, each
    ``{"role": "user"|"assistant", "content": str}``.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError(
                "history turns must be {'role': 'user'|'assistant', "
                "'content': str}"
            )
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_prompt(question, history=None):
    """Render the messages into a single chat prompt (no model required)."""
    parts = []
    for message in build_messages(question, history):
        parts.append(
            "<|im_start|>{}\n{}<|im_end|>".format(
                message["role"], message["content"]
            )
        )
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def generate(question, history=None, model_id=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS):
    """Run Spark-X2.6-4B locally and return the plain-language answer."""
    messages = build_messages(question, history)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "generate() needs torch and transformers. Install them and make "
            "sure {} is downloadable, or run "
            "`python health_chatbot.py --dry-run` to inspect the prompt.".format(
                model_id
            )
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    completion = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B plain-language health information demo"
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the prompt that would be sent, without loading the model",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        print(build_prompt(args.question))
        return 0

    print(generate(args.question, max_new_tokens=args.max_new_tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
