"""Minimal Spark-X2.6-4B health-information chatbot.

Demo application for HER Hack-Astron #7 (AI for Good): an inclusive,
plain-language health-education chatbot. It answers general health
questions and never diagnoses, prescribes, or replaces a clinician.

The heavy dependencies (`torch`, `transformers`) are imported lazily, so
this module can be imported and unit-tested without downloading weights.
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain general health topics in "
    "plain language for readers with no medical training.\n"
    "Rules:\n"
    "1. Never diagnose, never state a likely disease as a conclusion, and never "
    "prescribe or adjust any medication or dose.\n"
    "2. Give only general, well-established information; say clearly when you "
    "are unsure or when the topic needs a professional.\n"
    "3. Always tell the reader to consult a qualified clinician about their own "
    "situation.\n"
    "4. List clear red-flag warning signs that mean 'seek urgent care now'.\n"
    "5. Do not ask for or repeat personal identifying details.\n"
    "6. Keep answers short: a few short paragraphs or bullet points."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days, starting two days after "
    "I came back from a trip. What general information should I know, and when "
    "should I see a doctor?"
)

DISCLAIMER = (
    "This is general health education from a language model, not medical advice, "
    "diagnosis, or treatment. Please consult a qualified clinician."
)


def build_messages(
    question: str,
    history: Optional[Iterable[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt is always first and the current question is always
    last. Unknown roles and empty turns in `history` are dropped.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def format_reply(text: str) -> str:
    """Attach the safety disclaimer exactly once."""
    body = (text or "").strip()
    if not body:
        return DISCLAIMER
    if DISCLAIMER.lower() in body.lower():
        return body
    return body + "\n\n---\n" + DISCLAIMER


def generate(
    question: str,
    history: Optional[Iterable[Dict[str, str]]] = None,
    model_id: str = MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Run one Spark-X2.6-4B generation locally and return the reply text."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    messages = build_messages(question, history)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )

    if getattr(tokenizer, "chat_template", None):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
    else:
        prompt = "\n\n".join(
            "{0}: {1}".format(m["role"].capitalize(), m["content"]) for m in messages
        )
        prompt += "\n\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    generated = output[0][inputs["input_ids"].shape[-1]:]
    return format_reply(tokenizer.decode(generated, skip_special_tokens=True))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B plain-language health information chatbot (demo)."
    )
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
    parser.add_argument("--model-id", default=MODEL_ID, help="model id or local path")
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="generation budget"
    )
    args = parser.parse_args(argv)

    print("Spark-X2.6-4B health information demo")
    print("Model: {0}".format(args.model_id))
    print("Question: {0}".format(args.question))
    print("-" * 60)
    print(generate(args.question, model_id=args.model_id, max_new_tokens=args.max_new_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
