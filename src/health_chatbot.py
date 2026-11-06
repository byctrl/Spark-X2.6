"""Minimal Spark-X2.6-4B health-information chatbot.

HER Hack-Astron #7 entry: inclusive access to plain-language health
information. The model only supports patient education; it does not
diagnose, prescribe or replace a qualified clinician.

The heavy runtime (torch / transformers) is imported lazily inside
``generate_reply`` so that importing this module stays cheap and the
unit tests can run without downloading model weights.
"""

from __future__ import annotations

from typing import Dict, List

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who may have limited "
    "health literacy. Answer in plain language, short sentences and everyday "
    "words. Explain general background only. Never diagnose, never name a "
    "prescription medicine, never give a dose, and never tell the user to stop "
    "or change a treatment. If the situation sounds urgent, say clearly that "
    "the person should contact emergency services or a clinician now. Always "
    "end by reminding the user to confirm anything important with a qualified "
    "health professional."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days, and I came back from a "
    "trip abroad last week. I do not have a doctor here. What should I know "
    "before I decide what to do?"
)

DISCLAIMER = (
    "This is general education, not medical advice. Please confirm anything "
    "important with a qualified health professional."
)


def build_messages(question: str) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_prompt(question: str) -> str:
    """Render the messages as a single plain-text prompt."""
    lines = []
    for message in build_messages(question):
        lines.append("{0}: {1}".format(message["role"].capitalize(), message["content"]))
    lines.append("Assistant:")
    return "\n\n".join(lines)


def clean_reply(text: str) -> str:
    """Normalise model output and make sure the safety reminder is present."""
    cleaned = (text or "").strip()
    if not cleaned:
        return DISCLAIMER
    if "health professional" not in cleaned.lower():
        cleaned = cleaned + "\n\n" + DISCLAIMER
    return cleaned


def load_pipeline():
    """Load the Spark-X2.6-4B text-generation pipeline (lazy import)."""
    from transformers import pipeline

    return pipeline("text-generation", model=MODEL_ID, device_map="auto")


def generate_reply(question: str, generator=None) -> str:
    """Return a plain-language reply for ``question``.

    ``generator`` can be injected for testing; when omitted the real
    Spark-X2.6-4B pipeline is loaded on first use.
    """
    prompt = build_prompt(question)
    if generator is None:
        generator = load_pipeline()
    output = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
    )
    if isinstance(output, list):
        output = output[0]
    if isinstance(output, dict):
        text = output.get("generated_text", "")
    else:
        text = str(output)
    return clean_reply(text)


def main() -> None:
    print("Spark-X2.6-4B health information demo")
    print("Model: {0}".format(MODEL_ID))
    print("-" * 60)
    print("Question: {0}".format(DEMO_QUESTION))
    print("-" * 60)
    print(generate_reply(DEMO_QUESTION))


if __name__ == "__main__":
    main()
