"""Minimal Spark-X2.6-4B health-knowledge chatbot for the AI for Good challenge.

The demo answers general patient-education questions in plain language. It is
not a medical device: it refuses diagnosis and treatment advice and always
points the user to a qualified clinician.
"""

from __future__ import annotations

import argparse

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people with limited access to health "
    "information. Explain general health topics in plain, simple language. "
    "Do not diagnose, do not name a likely disease for the user, and do not recommend "
    "medicines, dosages or treatments. Do not claim to replace a clinician. "
    "Always tell the user to consult a qualified health worker, and clearly flag "
    "symptoms that may need urgent care. Never ask for names, addresses or other "
    "identifying details."
)

DEMO_QUESTION = (
    "I returned from a trip two weeks ago and since then I have a cough and a mild "
    "fever in the evening. What should I know before I see a doctor?"
)


def build_messages(question, language="en"):
    """Return the chat messages sent to Spark-X2.6-4B for one question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    language = (language or "en").strip() or "en"
    system = SYSTEM_PROMPT + " Reply in language code: " + language + "."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def load_pipeline():
    """Load Spark-X2.6-4B through transformers (optional dependency)."""
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "transformers is required to run the model: "
            "pip install transformers accelerate"
        ) from exc
    return pipeline("text-generation", model=MODEL_ID, device_map="auto")


def extract_reply(output):
    """Pull the assistant text out of a transformers text-generation output."""
    if not output:
        return ""
    first = output[0]
    generated = first.get("generated_text", first) if isinstance(first, dict) else first
    if isinstance(generated, list):
        for message in reversed(generated):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", "")).strip()
        return ""
    return str(generated).strip()


def answer(question, language="en", pipe=None):
    """Generate one plain-language answer with Spark-X2.6-4B."""
    messages = build_messages(question, language)
    pipe = load_pipeline() if pipe is None else pipe
    output = pipe(messages, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    reply = extract_reply(output).strip()
    if not reply:
        raise RuntimeError("the model returned an empty reply")
    return reply


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a general health-education question."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="your question; defaults to DEMO_QUESTION when omitted",
    )
    parser.add_argument("--language", default="en", help="reply language code")
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip() or DEMO_QUESTION
    print(answer(question, language=args.language))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
