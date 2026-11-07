"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

Goal: give people with limited access to a clinician a plain-language way to
understand general health information. The model never diagnoses, never
prescribes, and always points the user to a professional.

The model is used for what it is good at: turning a plain question into a
short, simple-language explanation, listing warning signs that need urgent
care, and telling the user what to ask a clinician.

Run the built-in demo (downloads XHToken/Spark-X2.6-4B, needs torch + transformers):

    python src/health_chatbot.py --demo

Interactive chat:

    python src/health_chatbot.py

The module imports without torch/transformers installed, so the tests run
anywhere.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, List

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "It cannot diagnose or treat you. Please consult a qualified clinician."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who may have limited "
    "access to healthcare. Follow these rules without exception:\n"
    "1. Use simple, everyday language and short paragraphs.\n"
    "2. Explain general health concepts and what a symptom usually means.\n"
    "3. Never diagnose, never name a prescription medicine or a dose, and "
    "never tell the user to start or stop a treatment.\n"
    "4. Always point the user to a qualified clinician, and say clearly when a "
    "symptom needs urgent care (for example chest pain, trouble breathing, "
    "confusion, heavy bleeding, or a fever that does not go away).\n"
    "5. If the question is not about health, say so briefly and redirect.\n"
    "6. Do not ask for or repeat names, addresses, or other identifying details."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a cough and some "
    "fever in the evenings. There is no doctor near me. What should I know?"
)


def build_messages(
    question: str,
    history: Iterable[Dict[str, str]] | None = None,
) -> List[Dict[str, str]]:
    """Build the chat messages that are sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    for turn in history or ():
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


class HealthChatbot:
    """Thin wrapper around a local Spark-X2.6-4B text-generation pipeline."""

    def __init__(self, model_id: str = MODEL_ID, max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def load(self) -> "HealthChatbot":
        """Load the model lazily so importing this module stays cheap."""
        try:
            from transformers import pipeline  # noqa: WPS433 (lazy import)
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "transformers and torch are required to run the model. "
                "Install them and download XHToken/Spark-X2.6-4B first."
            ) from exc

        self._pipe = pipeline(
            "text-generation",
            model=self.model_id,
            device_map="auto",
        )
        return self

    def ask(
        self,
        question: str,
        history: Iterable[Dict[str, str]] | None = None,
    ) -> str:
        """Answer one question and append the fixed safety disclaimer."""
        messages = build_messages(question, history)
        if self._pipe is None:
            self.load()

        output = self._pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            text = generated[-1].get("content", "")
        else:
            text = str(generated)

        text = text.strip() or "I could not produce an answer. Please ask a clinician."
        return "{0}\n\n---\n{1}".format(text, DISCLAIMER)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health-information chatbot")
    parser.add_argument("--model", default=MODEL_ID, help="model id or local path")
    parser.add_argument("--demo", action="store_true", help="ask the built-in demo question and exit")
    parser.add_argument("--question", default=None, help="ask a single question and exit")
    args = parser.parse_args(argv)

    bot = HealthChatbot(model_id=args.model)

    if args.demo:
        print("Q:", DEMO_QUESTION)
        print()
        print(bot.ask(DEMO_QUESTION))
        return 0

    if args.question:
        print(bot.ask(args.question))
        return 0

    print("Spark-X2.6-4B health-information chatbot. Type 'quit' to exit.")
    print(DISCLAIMER)
    history: List[Dict[str, str]] = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in ("quit", "exit"):
            return 0
        if not question:
            continue
        answer = bot.ask(question, history)
        print("\nAssistant:", answer)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    sys.exit(main())
