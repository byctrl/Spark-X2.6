#!/usr/bin/env python3
"""Minimal Spark-X2.6-4B demo: inclusive plain-language health information chatbot.

Scope: patient *education* only. The assistant never diagnoses, prescribes or
replaces a clinician; it explains things in plain language and points people to
professional care. This keeps the demo inside the AI-for-Good brief (health and
accessibility) while staying a small, reviewable application.

Usage:
    python src/health_chatbot.py                  # run the built-in demo question
    python src/health_chatbot.py "my question"    # ask your own question

Running the demo requires torch + transformers and the model weights
XHToken/Spark-X2.6-4B downloaded from Hugging Face. Message building and prompt
formatting use only the standard library, so they can be tested without the model.
"""

from __future__ import annotations

import argparse
import json
import sys

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in plain, "
    "respectful language that a person without medical training can understand. "
    "You do NOT diagnose, prescribe, or recommend specific treatments or dosages. "
    "Always say clearly when a symptom needs professional or emergency care, and "
    "encourage the person to consult a qualified clinician. Do not ask for or repeat "
    "identifying personal data. If the question is outside health education, say so "
    "briefly and suggest a better source."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough and "
    "some fever in the evenings. What could this mean, and when should I see a doctor?"
)


def build_messages(question, history=None):
    """Return chat messages for Spark-X2.6: system prompt + optional history + question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_prompt(messages, tokenizer=None):
    """Format messages with the tokenizer chat template, or a readable fallback."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # older tokenizers without a chat template
            pass

    lines = ["<|{0}|>\n{1}".format(m["role"], m["content"]) for m in messages]
    lines.append("<|assistant|>\n")
    return "\n".join(lines)


def load_model(model_id=MODEL_ID):
    """Load the tokenizer and causal LM. Imported lazily so tests need no deps."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    model.eval()
    return tokenizer, model


def generate_reply(question, history=None, tokenizer=None, model=None, max_new_tokens=MAX_NEW_TOKENS):
    """Generate one plain-language answer with Spark-X2.6-4B."""
    import torch

    messages = build_messages(question, history)
    if tokenizer is None or model is None:
        tokenizer, model = load_model()

    prompt = build_prompt(messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    new_tokens = output[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health information chatbot")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION, help="health question to ask")
    parser.add_argument("--model", default=MODEL_ID, help="model id or local path")
    parser.add_argument("--json", action="store_true", help="print messages and reply as JSON")
    args = parser.parse_args(argv)

    tokenizer, model = load_model(args.model)
    reply = generate_reply(args.question, tokenizer=tokenizer, model=model)

    if args.json:
        print(json.dumps({
            "model": args.model,
            "messages": build_messages(args.question),
            "reply": reply,
        }, ensure_ascii=False, indent=2))
    else:
        print("Q:", args.question)
        print()
        print(reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
