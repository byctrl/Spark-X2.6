"""Minimal Spark-X2.6-4B health-information chatbot.

Public-interest demo for the HER Hack-Astron #7 Spark-X2.6 AI for Good
challenge. The assistant explains health topics in plain language, never
diagnoses or prescribes, and always redirects the reader to a qualified
clinician.

Run the illustrated demo from the repository root:

    python src/health_chatbot.py "I have had a cough and fever for three days"

The model is loaded lazily by :func:`_default_generate`, so importing this
module (and running the tests) does not require downloading weights.
"""

from __future__ import annotations

import sys
from typing import Callable, Iterable, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
MAX_INPUT_CHARS = 2000

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health topics in simple, "
    "respectful language that a person without medical training can follow. "
    "You must NOT diagnose, prescribe, or replace a clinician. Do not give "
    "dosages or treatment plans. If the question suggests an emergency "
    "(chest pain, difficulty breathing, severe bleeding, fainting, sudden "
    "weakness, or a symptom in a pregnant person or an infant), tell the user "
    "to contact emergency services or a clinician immediately. Always end "
    "with a short reminder to consult a qualified health professional."
)

DEMO_QUESTION = (
    "I have had a cough and mild fever since I got back from a trip "
    "three days ago. What should I know?"
)

SAFETY_NOTICE = (
    "This is general health information, not medical advice. "
    "Please consult a qualified health professional about your own situation."
)

FALLBACK_REPLY = (
    "I could not produce an answer for that question. "
    "Please try rephrasing it."
)

_EMPTY_REPLY_MARKER = "could not produce an answer"


def build_messages(
    question: str,
    history: Optional[Iterable[dict]] = None,
) -> List[dict]:
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt always comes first, history turns are kept in order
    (only ``user``/``assistant`` roles), and the current question is last.
    Oversized questions are truncated rather than rejected so that a long
    pasted notice still produces an answer.
    """
    text = (question or "").strip()
    if not text:
        raise ValueError("question must be a non-empty string")
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})
    return messages


def build_prompt(messages: Iterable[dict]) -> str:
    """Render messages with plain role markers for the text-generation call."""
    parts = [f"<|{m['role']}|>\n{m['content']}\n" for m in messages]
    parts.append("<|assistant|>\n")
    return "".join(parts)


def _load_pipeline():
    """Load the Spark-X2.6-4B text-generation pipeline (lazy, heavy)."""
    from transformers import pipeline  # imported lazily on purpose

    return pipeline("text-generation", model=MODEL_ID, device_map="auto")


def _default_generate(messages: List[dict], max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate a reply with Spark-X2.6-4B."""
    generator = _load_pipeline()
    prompt = build_prompt(messages)
    output = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    text = output[0]["generated_text"]
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def finalize(raw_reply: str) -> str:
    """Guarantee a non-empty reply that carries the safety notice."""
    reply = (raw_reply or "").strip()
    if not reply or _EMPTY_REPLY_MARKER in reply:
        reply = FALLBACK_REPLY
    if SAFETY_NOTICE not in reply:
        reply = f"{reply}\n\n{SAFETY_NOTICE}"
    return reply


def answer(
    question: str,
    history: Optional[Iterable[dict]] = None,
    generate: Optional[Callable[..., str]] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Answer a health question with Spark-X2.6-4B.

    ``generate`` is injectable so the demo logic can be tested without a GPU
    or a model download; it receives the message list and ``max_new_tokens``.
    """
    messages = build_messages(question, history)
    generator = generate or _default_generate
    raw = generator(messages, max_new_tokens=max_new_tokens)
    return finalize(raw)


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or DEMO_QUESTION
    print(f"Q: {question}\n")
    print(answer(question))


if __name__ == "__main__":
    main()
