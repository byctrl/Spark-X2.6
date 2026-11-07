"""Minimal health-information chatbot built on Spark-X2.6-4B.

Public-interest demo for the HER Hack-Astron #7 "AI for Good" challenge:
a plain-language patient-education assistant. It explains health topics and
points people to a qualified clinician; it never diagnoses or prescribes.

Usage:
    python src/health_chatbot.py             # runs DEMO_QUESTION
    python src/health_chatbot.py "my question"

Requires torch and transformers plus the model XHToken/Spark-X2.6-4B
(downloaded from Hugging Face or already present in the local HF cache).
"""

from __future__ import annotations

import sys
from typing import Dict, List

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health topics in plain, "
    "simple language that a person without medical training can understand. "
    "Rules you must always follow:\n"
    "1. Do not diagnose, do not name a likely disease as the user's condition, "
    "and do not recommend prescription drugs, dosages, or treatments.\n"
    "2. Do not claim to replace a doctor. Always tell the user to consult a "
    "qualified clinician for their own situation.\n"
    "3. If the message suggests an emergency (for example chest pain, "
    "difficulty breathing, heavy bleeding, fainting, or thoughts of self-harm), "
    "tell the user to contact local emergency services immediately.\n"
    "4. Say clearly when you are unsure, and never invent sources, statistics, "
    "or guidelines.\n"
    "5. Respect privacy: do not ask for names, ID numbers, addresses, or other "
    "identifying details."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and mild fever in the evening. What could it be and when should I see a "
    "doctor?"
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "Please consult a qualified clinician about your own situation."
)


# ---------------------------------------------------------------------------
# Prompt construction (pure, no model needed)
# ---------------------------------------------------------------------------
def build_messages(
    question: str, system_prompt: str = SYSTEM_PROMPT
) -> List[Dict[str, str]]:
    """Build the chat messages that are sent to the model."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def format_prompt(messages: List[Dict[str, str]]) -> str:
    """Render the messages into a plain chat prompt for the tokenizer."""
    parts = []
    for message in messages:
        parts.append("<|{role}|>\n{content}".format(**message))
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def load_pipeline(model_id: str = MODEL_ID):
    """Load the Spark-X2.6 text-generation pipeline (heavy import is lazy)."""
    from transformers import pipeline

    return pipeline("text-generation", model=model_id, device_map="auto")


def _extract_text(outputs, prompt: str) -> str:
    """Pull the generated text out of the pipeline output, dropping any echo."""
    if not outputs:
        return ""
    item = outputs[0]
    text = item.get("generated_text", "") if isinstance(item, dict) else str(item)
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def answer(
    question: str,
    generator=None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Return a plain-language, safety-bounded answer to a health question."""
    prompt = format_prompt(build_messages(question))
    if generator is None:
        generator = load_pipeline()
    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    text = _extract_text(outputs, prompt)
    if not text:
        text = "I could not produce a reliable answer. Please ask a clinician."
    if DISCLAIMER not in text:
        text = "{}\n\n{}".format(text, DISCLAIMER)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print("Question: {}\n".format(question))
    print(answer(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
