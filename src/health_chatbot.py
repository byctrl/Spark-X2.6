"""Minimal health-information chatbot powered by Spark-X2.6-4B.

This demo is for public-interest health education. It does not diagnose or
prescribe. Run it locally after downloading the model:

    python src/health_chatbot.py --question "..."

The model is loaded lazily so that helper functions and tests do not require
transformers or model weights.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in "
    "plain language. Do not diagnose, prescribe, or replace a clinician. "
    "State uncertainty clearly, mention urgent warning signs when relevant, "
    "and advise the user to contact a qualified healthcare professional. "
    "Never ask for or repeat unnecessary personal identifiers."
)

DEMO_QUESTION = (
    "I returned from a trip three days ago and now have a cough and fever. "
    "What questions should I ask a doctor, and when should I seek urgent care?"
)


def build_messages(question: str, history: Optional[Sequence[dict]] = None) -> List[dict]:
    """Build a chat message list with the fixed safety system prompt."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for item in history:
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"}:
                raise ValueError("history roles must be 'user' or 'assistant'")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("history content must be non-empty text")
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def generate_reply(
    question: str,
    history: Optional[Sequence[dict]] = None,
    model=None,
    tokenizer=None,
) -> str:
    """Generate a reply with Spark-X2.6-4B.

    Pass model/tokenizer for tests or custom loading. Otherwise they are loaded
    from MODEL_ID using transformers. Requires a local runtime with the model
    available (or Hugging Face access configured by the caller).
    """
    messages = build_messages(question, history)
    if model is None or tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
        )
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    input_length = inputs["input_ids"].shape[-1]
    reply_ids = output_ids[0][input_length:]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health-information chatbot")
    parser.add_argument("--question", default=DEMO_QUESTION, help="Question to ask")
    args = parser.parse_args(argv)
    print(generate_reply(args.question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
