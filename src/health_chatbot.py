"""Minimal health-information chatbot built on Spark-X2.6-4B.

Purpose (HER Hack-Astron #7, AI for Good):
    Give people plain-language, non-diagnostic health information so they can
    prepare for a real appointment, even when they have no medical background.

What the model does here:
    ``XHToken/Spark-X2.6-4B`` is a text-only instruct model. It receives the
    chat messages built below and writes the assistant reply. The application
    keeps ownership of the safety rules: it fixes the system prompt, forbids
    diagnosis/treatment advice, and appends a constant disclaimer to every
    answer. Model output is never presented as ground truth or as a diagnosis.

The model is not a doctor and this file is not a medical device.

Usage::

    python src/health_chatbot.py                 # runs the demo question
    python src/health_chatbot.py "my question"   # asks your own question

The Hugging Face weights are only loaded when ``generate_reply`` is called
without a generator, so the prompt helpers stay testable on a machine that
has neither ``transformers`` nor a GPU installed.
"""

from __future__ import annotations

import argparse

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in plain, "
    "simple language for a reader with no medical training. "
    "Do not diagnose, do not suggest medicines or dosages, and never tell the user "
    "to start, stop or change a treatment. "
    "When a question needs a diagnosis or a treatment decision, say clearly that a "
    "qualified clinician has to decide, and list what the user should note down or "
    "bring to the appointment (symptoms, dates, medicines, travel, test results). "
    "Say when you are unsure, or when the information may not apply to this user. "
    "Never claim to be a doctor and never replace emergency services: if the "
    "situation may be urgent, tell the user to contact local emergency care now. "
    "Answer in the language the user writes in."
)

DEMO_QUESTION = (
    "I have had a dry cough and a mild fever for four days and I travelled last "
    "week. What should I keep track of before I see a doctor?"
)

DISCLAIMER = (
    "\n\n---\n"
    "This is general education only, not a diagnosis or a treatment plan. "
    "Please confirm anything important with a qualified health professional; "
    "if your symptoms feel urgent, contact local emergency care now."
)


def build_messages(question, history=None):
    """Return the chat messages that are sent to Spark-X2.6-4B."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            raise ValueError("history turns need role 'user'/'assistant' and content")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})
    return messages


def format_prompt(messages, tokenizer=None):
    """Render messages with the model chat template, or a plain fallback."""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    lines = [f"{message['role']}: {message['content']}" for message in messages]
    lines.append("assistant:")
    return "\n\n".join(lines)


def add_disclaimer(text):
    """Attach the fixed safety notice exactly once."""
    body = (text or "").strip()
    if not body:
        raise ValueError("model returned an empty answer")
    if DISCLAIMER.strip() in body:
        return body
    return body + DISCLAIMER


def load_generator(model_id=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS):
    """Load Spark-X2.6-4B with transformers (imported lazily)."""
    from transformers import pipeline

    return pipeline("text-generation", model=model_id, max_new_tokens=max_new_tokens)


def _extract_text(result):
    """Pull the assistant text out of a transformers text-generation result."""
    first = result[0] if isinstance(result, list) and result else result
    if isinstance(first, dict):
        first = first.get("generated_text", "")
    if isinstance(first, list):
        for message in reversed(first):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return (message.get("content") or "").strip()
        return ""
    return str(first or "").strip()


def generate_reply(question, history=None, generator=None, max_new_tokens=MAX_NEW_TOKENS):
    """Answer one question. Pass ``generator`` to avoid loading the model."""
    messages = build_messages(question, history)
    if generator is None:
        generator = load_generator(max_new_tokens=max_new_tokens)
    result = generator(messages, max_new_tokens=max_new_tokens, do_sample=False)
    return add_disclaimer(_extract_text(result))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information demo (education only)"
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="question to ask; defaults to the built-in demo question",
    )
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip() or DEMO_QUESTION

    print(f"Q: {question}\n")
    try:
        print(generate_reply(question))
    except ValueError as exc:
        print(f"No answer produced: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
