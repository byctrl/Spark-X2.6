"""Minimal health-knowledge chatbot demo for HER Hack-Astron #7 (Issue #16).

Built on XHToken/Spark-X2.6-4B. The app answers general health questions in
plain language for people who cannot easily reach a clinician. It is an
*education* assistant: it never diagnoses, never prescribes, and always tells
the reader to confirm with a qualified professional.

Usage::

    python src/health_chatbot.py                 # run the built-in demo question
    python src/health_chatbot.py "my question"   # ask your own question

The model is loaded lazily, so importing this module (and running the tests)
does not require torch/transformers or a GPU.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant on a public health helpline. "
    "Explain health topics in simple, everyday language (about a 9th-grade "
    "reading level) and keep answers short and concrete.\n"
    "Rules you must always follow:\n"
    "1. You do NOT diagnose and you do NOT prescribe or change medication.\n"
    "2. Do not present a likely disease as a conclusion.\n"
    "3. Name the warning signs that mean the person should get urgent care.\n"
    "4. Tell the reader to confirm with a qualified clinician.\n"
    "5. Never ask for or repeat personal identifiers.\n"
    "If the question is unclear, say what general information would help."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have had a dry "
    "cough and a fever in the evenings. What should I know before I see a doctor?"
)

SAFETY_FOOTER = (
    "This is general health education, not a diagnosis and not a treatment "
    "plan. Please confirm anything important with a qualified clinician."
)


def build_messages(question: str) -> List[Dict[str, str]]:
    """Wrap a user question in the system/user message pair sent to Spark."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def build_prompt(tokenizer, question: str) -> str:
    """Render the messages with the tokenizer's chat template when available."""
    messages = build_messages(question)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            return apply_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    rendered = "\n\n".join(
        "{0}: {1}".format(message["role"].upper(), message["content"])
        for message in messages
    )
    return rendered + "\n\nASSISTANT:"


def format_output(reply: str) -> str:
    """Ensure every answer shown to a user carries the safety footer."""
    text = (reply or "").strip()
    if "qualified clinician" in text.lower():
        return text
    if not text:
        text = "I could not produce an answer. Please try rephrasing the question."
    return text + "\n\n" + SAFETY_FOOTER


def load_model(model_id: str = MODEL_ID):
    """Load the Spark-X2.6-4B tokenizer and model (requires torch + transformers)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def answer(
    question: str,
    tokenizer=None,
    model=None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate a plain-language answer with Spark-X2.6-4B."""
    prompt = build_prompt(tokenizer, question) if tokenizer is not None else None
    if tokenizer is None or model is None:
        tokenizer, model = load_model()
    if prompt is None:
        prompt = build_prompt(tokenizer, question)

    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    reply = answer(question)
    print("Q: " + question)
    print("")
    print(format_output(reply))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
