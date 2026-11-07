"""Minimal health-information chatbot built on Spark-X2.6-4B.

HER Hack-Astron #7 — AI for Good entry (health / accessibility).

Scope and safety boundaries:
* The model is used only for plain-language patient *education*.
* The system prompt forbids diagnosis, prescriptions and dosage advice and
  redirects the user to a qualified clinician.
* The application never presents model output as medical advice; the
  disclaimer is printed together with every answer.

Nothing here needs a GPU to import: ``transformers``/``torch`` are imported
lazily inside :func:`load_model`, so the helpers and the test suite run on a
plain Python install.

Usage::

    python src/health_chatbot.py                     # demo question
    python src/health_chatbot.py -q "my question"    # custom question
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "Please consult a qualified clinician about your own situation."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who may have limited "
    "access to healthcare. Answer in plain language: short sentences, no "
    "jargon. You must NOT diagnose, prescribe, recommend specific medicines "
    "or dosages, or replace a clinician. Explain in general terms what the "
    "described symptoms may relate to, list warning signs that need urgent "
    "care, and tell the user to contact a qualified health worker. If the "
    "question suggests an emergency, say clearly that emergency services "
    "should be contacted now. Keep the answer under 200 words and end with "
    "one practical next step."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and a mild fever in the evenings. There is no doctor nearby. "
    "What should I know, and when should I seek care?"
)


def build_messages(
    question: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """Return the chat messages sent to the model for ``question``."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]


def format_prompt(messages: List[Dict[str, str]]) -> str:
    """Render messages as a role-tagged prompt ending with an assistant turn."""
    parts = [
        "<|{role}|>\n{content}\n".format(role=message["role"], content=message["content"])
        for message in messages
    ]
    parts.append("<|assistant|>\n")
    return "".join(parts)


def load_model(model_id: str = MODEL_ID):
    """Load the Spark-X2.6-4B tokenizer and model (heavy imports are lazy)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate(
    question: str,
    model=None,
    tokenizer=None,
    model_id: str = MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Answer ``question`` with Spark-X2.6-4B, loading it lazily if needed."""
    messages = build_messages(question)
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_id)

    inputs = tokenizer(format_prompt(messages), return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    prompt_len = len(inputs["input_ids"][0])
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_tokens = outputs[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Health-information chatbot powered by " + MODEL_ID,
    )
    parser.add_argument("-q", "--question", default=DEMO_QUESTION)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    answer = generate(
        args.question,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )
    print("Question: " + args.question + "\n")
    print("Answer: " + answer + "\n")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
