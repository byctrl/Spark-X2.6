"""Minimal health-information chatbot built on Spark-X2.6-4B.

Created for the HER Hack-Astron #7 "AI for Good" challenge (Issue #16),
health / accessibility track.

What the model does in this application
---------------------------------------
``XHToken/Spark-X2.6-4B`` is the single generation backend. It is a text-only
instruction-tuned model, so this script:

1. wraps the user's plain-language health question in a bounded system prompt,
2. asks the model to explain the topic in simple language, name the red-flag
   signs that mean "seek emergency care now", and suggest what to prepare for
   a clinician visit,
3. appends a fixed disclaimer that the model never diagnoses or prescribes.

The model never replaces a clinician: the output is patient education only.

Usage
-----
    python src/health_chatbot.py --dry-run
    python src/health_chatbot.py "I have had a cough and fever for four days."
    python src/health_chatbot.py --json "..."

``--dry-run`` prints the exact prompt without loading the model, so the demo is
reproducible on machines without a GPU.
"""

from __future__ import annotations

import argparse
import json

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

ALLOWED_ROLES = ("user", "assistant")

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public health help desk. "
    "Explain health information in plain language at about a 6th-grade reading level. "
    "You do NOT diagnose, you do NOT prescribe, you do NOT give medicine doses, and you "
    "never tell the user to start, stop or change a treatment a clinician prescribed. "
    "For every answer: (1) briefly restate what the user described, (2) give general, "
    "non-prescriptive information about when to seek care and what to prepare for the "
    "appointment, (3) list the red-flag signs that mean emergency care is needed now, "
    "(4) remind the user that this is general education and encourage them to contact a "
    "qualified clinician. "
    "If the user describes an emergency (chest pain, trouble breathing, severe bleeding, "
    "signs of stroke, thoughts of self-harm, or a pregnancy emergency), your FIRST sentence "
    "must tell them to contact emergency services now. "
    "If you are unsure, say that you are unsure. Never invent sources, statistics, "
    "medicine names, doses, or local phone numbers."
)

DISCLAIMER = (
    "This is general health education, not medical advice. It cannot diagnose you. "
    "Please contact a qualified clinician about your own situation."
)

DEMO_QUESTION = (
    "I came back from a trip last week and now I have a cough and a fever of 38.5C."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    ``history`` is an optional list of earlier ``{"role", "content"}`` turns;
    it is copied, never mutated. A blank question is rejected so the model is
    never called with an empty prompt.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError("history role must be one of %s" % (ALLOWED_ROLES,))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("history content must be a non-empty string")
        messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_prompt(messages, tokenizer=None):
    """Render messages to text, using the model chat template when available."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    parts = ["<|%s|>\n%s" % (m["role"], m["content"]) for m in messages]
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def ensure_disclaimer(text):
    """Append the fixed disclaimer once, even if the model already added one."""
    body = (text or "").strip()
    if DISCLAIMER.lower() in body.lower():
        return body
    if not body:
        return DISCLAIMER
    return body + "\n\n" + DISCLAIMER


def extract_reply(output):
    """Pull the assistant text out of a transformers pipeline result."""
    if isinstance(output, dict):
        output = [output]
    if not output:
        return ""

    item = output[0]
    if isinstance(item, dict):
        item = item.get("generated_text", "")
    if isinstance(item, list):
        if not item:
            return ""
        last = item[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(item)


def load_pipeline(model_id=MODEL_ID):
    """Load Spark-X2.6-4B. Imported lazily so tests need no GPU or weights."""
    from transformers import pipeline

    return pipeline("text-generation", model=model_id, device_map="auto")


def generate(messages, pipe, max_new_tokens=MAX_NEW_TOKENS):
    """Run one deterministic, bounded generation and apply the disclaimer."""
    output = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
    return ensure_disclaimer(extract_reply(output))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot (education only)"
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the prompt without loading the model",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON object")
    args = parser.parse_args(argv)

    messages = build_messages(args.question)

    if args.dry_run:
        prompt = build_prompt(messages)
        reply = None
    else:
        pipe = load_pipeline(args.model)
        prompt = build_prompt(messages, tokenizer=getattr(pipe, "tokenizer", None))
        reply = generate(
            messages, pipe, max_new_tokens=args.max_new_tokens
        )

    if args.json:
        payload = {"model": args.model, "prompt": prompt, "reply": reply}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif reply is None:
        print(prompt)
    else:
        print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
