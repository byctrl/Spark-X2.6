#!/usr/bin/env python3
"""Minimal health-information chatbot powered by Spark-X2.6-4B.

Smallest useful "AI for Good" demo for the HER Hack-Astron #7 challenge:
a plain-language health education assistant. It does not diagnose, prescribe
or replace a clinician, and every answer points the user to professional care.

Usage:
    python src/health_chatbot.py "my question"
    python src/health_chatbot.py            # runs DEMO_QUESTION
"""

from __future__ import annotations

import argparse
import sys

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health topics in simple, "
    "respectful language that a person without medical training can follow. "
    "You must not diagnose, prescribe, or recommend specific medicines or doses. "
    "You must not claim to replace a doctor. Always say clearly when a situation "
    "may be urgent and tell the user to contact a qualified clinician or the "
    "local emergency service. If important information is missing, ask one short "
    "clarifying question instead of guessing."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough and "
    "a mild fever in the evening. What should I know, and when should I see a "
    "doctor? Please explain in simple words."
)

DISCLAIMER = (
    "This is general health education, not medical advice. Please contact a "
    "qualified clinician or your local emergency service for personal care."
)


def build_messages(question, system_prompt=SYSTEM_PROMPT):
    """Build the chat messages sent to the model for one user question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def render_prompt(messages, tokenizer=None):
    """Render messages with the tokenizer chat template, or a plain fallback."""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts = [
        "<|im_start|>%s\n%s<|im_end|>" % (m["role"], m["content"]) for m in messages
    ]
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def generate_reply(question, model=None, tokenizer=None, max_new_tokens=MAX_NEW_TOKENS):
    """Return the model answer text for one question.

    The heavy imports stay inside the function so the helpers above can be used
    (and tested) without installing transformers or downloading the weights.
    """
    messages = build_messages(question)
    if model is None or tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, device_map="auto"
        )
    prompt = render_prompt(messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    prompt_length = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def with_disclaimer(answer):
    """Always return an answer together with the safety disclaimer."""
    answer = (answer or "").strip()
    if not answer:
        return DISCLAIMER
    return answer + "\n\n" + DISCLAIMER


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot (demo)"
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    args = parser.parse_args(argv)
    print(with_disclaimer(generate_reply(args.question)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
