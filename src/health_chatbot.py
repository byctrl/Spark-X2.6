"""Minimal helpers for a Spark-X2.6-4B health-information chatbot.

This module intentionally has no third-party imports so that the helper
functions can be unit-tested without downloading model weights.
"""

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
MAX_QUESTION_CHARS = 2000

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in "
    "plain, simple language. You do not diagnose conditions, prescribe or "
    "recommend treatments, and you do not replace a qualified clinician. "
    "Encourage the user to seek professional care, especially for urgent or "
    "worsening symptoms. Be concise, avoid jargon, and ask a clarifying "
    "question when important details are missing."
)

DEMO_QUESTION = (
    "I have had a cough and a fever for three days since returning from a "
    "trip. What general information should I know, and when should I see a "
    "doctor?"
)


def validate_question(question):
    """Return a cleaned question or raise ValueError when invalid."""
    if not isinstance(question, str):
        raise ValueError("Question must be a string.")
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("Question must not be empty.")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise ValueError(
            "Question must be at most {} characters.".format(MAX_QUESTION_CHARS)
        )
    return cleaned


def build_messages(question):
    """Build an OpenAI-style chat message list for the model."""
    cleaned = validate_question(question)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": cleaned},
    ]
