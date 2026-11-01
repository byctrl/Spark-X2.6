"""Minimal accessible-health-information chatbot built on Spark-X2.6-4B.

Demo entry for HER Hack-Astron #7 (AI for Good). The application turns a
plain-language health question into a patient-education answer with a clear
safety boundary: it never diagnoses, never prescribes and always routes the
person to a qualified clinician. Everything runs locally.

Usage:
    python src/health_chatbot.py --dry-run
    python src/health_chatbot.py --question "..."
"""

from __future__ import annotations

import argparse

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in plain, "
    "respectful language that a non-specialist can follow, and keep answers short. "
    "Ask one clarifying question when the request is vague. "
    "Never diagnose, never prescribe, never give dosages, and never tell someone to "
    "start, stop or change a treatment. Never claim to replace a clinician. "
    "For anything that may be urgent, tell the person to contact a qualified "
    "clinician or a local emergency service right away, and name the kind of "
    "professional who can help. "
    "If the request is outside health education, say so and redirect. "
    "Reply in the language the person used."
)

DISCLAIMER = (
    "Educational information only - not medical advice, diagnosis or treatment. "
    "Contact a qualified clinician or your local emergency service for anything urgent."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for three days since I came back from a "
    "trip. What should I know before I see a doctor?"
)

_PIPE = None


def build_messages(question, history=None):
    """Build the chat messages for one turn, keeping prior turns if supplied."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError("history turns need role 'user'/'assistant' and string content")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def format_prompt(messages, tokenizer=None):
    """Use the model chat template when available, otherwise a plain fallback."""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts = []
    for message in messages:
        parts.append("<|{0}|>\n{1}\n".format(message["role"], message["content"]))
    parts.append("<|assistant|>\n")
    return "".join(parts)


def get_pipeline(model_id=MODEL_ID):
    """Load Spark-X2.6-4B once and reuse it for the rest of the session."""
    global _PIPE
    if _PIPE is None:
        from transformers import pipeline

        _PIPE = pipeline("text-generation", model=model_id, device_map="auto")
    return _PIPE


def _extract_text(output, prompt):
    if isinstance(output, str):
        text = output
    else:
        first = output[0]
        text = first["generated_text"] if isinstance(first, dict) else str(first)
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def generate_reply(question, history=None, generator=None, max_new_tokens=MAX_NEW_TOKENS):
    """Return the assistant answer for one question.

    ``generator`` may be injected (used by the tests) so the model never has to
    be downloaded to exercise the request-building and parsing logic.
    """
    messages = build_messages(question, history)
    gen = generator if generator is not None else get_pipeline()
    tokenizer = getattr(gen, "tokenizer", None)
    prompt = format_prompt(messages, tokenizer)
    output = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
    )
    return _extract_text(output, prompt)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health information chatbot")
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the rendered prompt without downloading the model",
    )
    args = parser.parse_args(argv)

    messages = build_messages(args.question)
    if args.dry_run:
        print(format_prompt(messages))
        return 0

    print(DISCLAIMER)
    print("")
    print(
        generate_reply(
            args.question,
            generator=get_pipeline(args.model),
            max_new_tokens=args.max_new_tokens,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
