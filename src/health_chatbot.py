"""Minimal Spark-X2.6-4B chatbot for inclusive health-knowledge access.

Public-interest demo for the HER Hack-Astron #7 challenge: explain health
information in plain language while refusing to diagnose or prescribe.

The module imports without transformers installed; the heavy runtime is only
loaded inside :func:`generate_answer`.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant helping people understand general "
    "health information. Answer in plain, simple language and short sentences, "
    "and define any medical term you must use. You do not diagnose conditions, "
    "do not interpret test results for a specific person, and do not recommend "
    "prescription medicines or dosages. Always encourage the reader to consult a "
    "qualified clinician for personal decisions, and say clearly that this is "
    "general educational information, not medical advice. If the question "
    "suggests an emergency (for example chest pain, difficulty breathing, heavy "
    "bleeding, fainting, or thoughts of self-harm), tell the reader to contact "
    "local emergency services right away before anything else. Do not ask for or "
    "repeat names, addresses, or other identifying details."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a cough and mild "
    "fever in the evenings. What could this be and what should I do next? I have "
    "no insurance and I am not sure where to go."
)

DISCLAIMER = (
    "\n\n---\nEducational information only, not medical advice. "
    "Please consult a qualified clinician about your own situation."
)


def build_messages(
    question: str,
    history: Optional[Sequence[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt is always first and the new user question is always last.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            raise ValueError("history turns need role 'user'/'assistant' and non-empty content")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_prompt(
    messages: Sequence[Dict[str, str]],
    tokenizer: Optional[object] = None,
) -> str:
    """Render messages with the model chat template, or a simple fallback."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
    lines = ["<|{0}|>\n{1}".format(m["role"], m["content"]) for m in messages]
    lines.append("<|assistant|>")
    return "\n".join(lines)


def with_disclaimer(answer: str) -> str:
    """Append the fixed educational-use notice to a model answer."""
    return (answer or "").strip() + DISCLAIMER


def generate_answer(
    question: str,
    model_id: str = MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
    history: Optional[Sequence[Dict[str, str]]] = None,
) -> str:
    """Run greedy decoding with Spark-X2.6-4B and return the answer text."""
    messages = build_messages(question, history)

    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    prompt = build_prompt(messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plain-language health knowledge chatbot on Spark-X2.6-4B."
    )
    parser.add_argument("question", nargs="?", default=None, help="question to ask")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo question")
    parser.add_argument("--model", default=MODEL_ID, help="model id or local path")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    question = args.question if args.question else DEMO_QUESTION
    print("Model:", args.model)
    print("Question:", question)
    print("Answer:")
    print(with_disclaimer(generate_answer(question, args.model, args.max_new_tokens)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
