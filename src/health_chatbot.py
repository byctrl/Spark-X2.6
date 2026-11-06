"""Minimal health-information chatbot for the Spark-X2.6-4B AI for Good challenge.

The model itself is text-only and has no medical authority. This module keeps the
application boundary deliberately small: a fixed patient-education system prompt,
a user question, and a disclaimer appended to every answer. Nothing here
diagnoses, prescribes, or replaces a clinician.

Run locally after downloading the model:

    python -m src.health_chatbot --question "..."
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health topics in short, "
    "plain sentences that someone without medical training can follow.\n"
    "Rules you must follow:\n"
    "1. Do not diagnose, do not present a likely disease as the answer, and do "
    "not prescribe or give doses for any medicine.\n"
    "2. Share general, widely accepted information and list the signs that mean "
    "the person should seek care urgently.\n"
    "3. Always tell the person to confirm with a qualified clinician or local "
    "health service, and state that this is not medical advice.\n"
    "4. If the question suggests an emergency (for example chest pain, trouble "
    "breathing, heavy bleeding, fainting, or thoughts of self-harm), say clearly "
    "that emergency help should be contacted now.\n"
    "5. Do not ask for names, addresses, or other identifying details."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago. For three days I have had a dry "
    "cough, a fever around 38.2 C and I feel very tired. What should I know, and "
    "when should I see a doctor?"
)

DISCLAIMER = (
    "This is general health education from an AI model, not medical advice. "
    "It cannot diagnose you. Please confirm with a qualified clinician or your "
    "local health service, and seek emergency care immediately if symptoms are severe."
)

Generator = Callable[[List[Dict[str, str]]], str]


class ChatError(RuntimeError):
    """Raised when the request or the model output cannot be used."""


def build_messages(question: str) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B for one question."""
    if not isinstance(question, str):
        raise ChatError("question must be a string")
    clean = question.strip()
    if not clean:
        raise ChatError("question must not be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": clean},
    ]


def _clean_reply(reply: Any) -> str:
    if isinstance(reply, dict):
        reply = reply.get("content", "")
    text = str(reply or "").strip()
    if not text:
        raise ChatError("the model returned an empty response")
    return text


def chat(question: str, generator: Optional[Generator] = None) -> str:
    """Return the model answer for ``question`` followed by the disclaimer."""
    messages = build_messages(question)
    generate = generator if generator is not None else load_generator()
    answer = _clean_reply(generate(messages))
    return f"{answer}\n\n{DISCLAIMER}"


def load_generator(model_id: str = MODEL_ID, max_new_tokens: int = MAX_NEW_TOKENS) -> Generator:
    """Load Spark-X2.6-4B with transformers and return a messages -> text callable."""
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ChatError(
            "transformers is not installed; install it and download "
            f"{model_id} from Hugging Face to run the demo"
        ) from exc

    pipe = pipeline("text-generation", model=model_id)

    def _generate(messages: List[Dict[str, str]]) -> str:
        outputs = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
        return outputs[0]["generated_text"][-1]["content"]

    return _generate


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health information chatbot")
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask the model")
    parser.add_argument("--model", default=MODEL_ID, help="model id to load")
    args = parser.parse_args(argv)

    try:
        print(chat(args.question, load_generator(args.model)))
    except ChatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
