"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

The module is dependency-free at import time: prompt building and the safety
rules can be tested without downloading model weights. The model itself is
loaded lazily inside generate_reply().
"""

from __future__ import annotations

import argparse
import sys

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
MAX_QUESTION_CHARS = 2000

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public health information service. "
    "Your job is to explain health topics in plain, everyday language so that people "
    "with low health literacy or limited access to care can understand them.\n"
    "Rules you must follow:\n"
    "1. Do not diagnose and do not prescribe. Never tell the person which disease they "
    "have and never recommend a medicine, dose, or treatment plan.\n"
    "2. Give general, well-established information only. If the question requires a "
    "personal medical decision, say so and refer the person to a qualified clinician.\n"
    "3. Say clearly when information is uncertain or when local guidance may differ.\n"
    "4. Describe red-flag signs that mean the person should seek urgent care now.\n"
    "5. Use short sentences and common words. Avoid medical jargon, or explain it.\n"
    "6. Never ask for or repeat identifying details such as names, addresses, or IDs.\n"
    "You are not a doctor and this conversation is not medical care."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a cough and a mild fever. "
    "I don't have a doctor nearby. What should I know, and when should I get help?"
)

DISCLAIMER = (
    "This is general education, not medical care. "
    "Please contact a qualified clinician about your own situation."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt is always first, prior valid turns are kept in order, and
    the current question is always last.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    if len(question) > MAX_QUESTION_CHARS:
        question = question[:MAX_QUESTION_CHARS].rstrip() + " [...]"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def format_prompt(messages, tokenizer=None):
    """Render messages with the tokenizer chat template, or a plain fallback."""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts = []
    for message in messages:
        parts.append("<|%s|>\n%s\n" % (message["role"], message["content"]))
    parts.append("<|assistant|>\n")
    return "".join(parts)


def load_model(model_id=MODEL_ID):
    """Load the Spark-X2.6-4B tokenizer and model (lazy import)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "transformers is required to run the model. Install it with "
            "`pip install transformers torch accelerate` and make sure "
            f"{model_id} is available locally or on the Hugging Face Hub."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return tokenizer, model


def generate_reply(
    question,
    history=None,
    tokenizer=None,
    model=None,
    max_new_tokens=MAX_NEW_TOKENS,
):
    """Return the model's plain-language answer to a health question."""
    messages = build_messages(question, history)
    if tokenizer is None or model is None:
        tokenizer, model = load_model()

    prompt = format_prompt(messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    generated = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health information chatbot (education only)"
    )
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="print the rendered prompt without loading the model",
    )
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    if args.prompt_only:
        print(format_prompt(build_messages(args.question)))
        return 0

    print(generate_reply(args.question, max_new_tokens=args.max_new_tokens))
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
