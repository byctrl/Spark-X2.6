"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Goal: give people plain-language health *education* without diagnosis or
prescription, and always route them back to a licensed clinician.

The model is loaded lazily, so importing this module (and running the unit
tests) never requires transformers, a GPU, or a network connection.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, Iterable, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in plain, "
    "kind language at roughly a 6th-grade reading level. You do not diagnose, "
    "prescribe, or recommend specific medicines or doses. Never tell the user to "
    "delay or stop professional care. When symptoms could be urgent, say so and "
    "direct the user to emergency services or a licensed clinician. Reply in the "
    "language the user writes in, keep the answer short and structured, and end by "
    "reminding the user that this is general education, not medical advice."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough and "
    "fever in the evening. What should I know before I see a doctor?"
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "Please consult a licensed clinician."
)

Generator = Callable[[str], str]


def build_messages(
    question: str,
    history: Optional[Iterable[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def format_prompt(messages: List[Dict[str, str]]) -> str:
    """Render messages with the ChatML markers Spark-X2.6 uses."""
    parts = [
        "<|im_start|>{role}\n{content}<|im_end|>".format(
            role=message["role"], content=message["content"]
        )
        for message in messages
    ]
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def ensure_disclaimer(text: str) -> str:
    """Guarantee exactly one safety disclaimer at the end of an answer."""
    text = (text or "").strip()
    if not text:
        return DISCLAIMER
    if DISCLAIMER in text:
        return text
    return text + "\n\n" + DISCLAIMER


class SparkHealthChatbot:
    """Thin, testable wrapper around a locally hosted Spark-X2.6-4B."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        max_new_tokens: int = MAX_NEW_TOKENS,
        generator: Optional[Generator] = None,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._generator = generator
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline  # lazy: not needed for unit tests

            self._pipeline = pipeline(
                "text-generation", model=self.model_id, device_map="auto"
            )
        return self._pipeline

    def _run_model(self, prompt: str) -> str:
        if self._generator is not None:
            return self._generator(prompt)
        pipe = self._load_pipeline()
        output = pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        if isinstance(output, list) and output and isinstance(output[0], dict):
            return output[0].get("generated_text", "") or ""
        return str(output)

    def generate(
        self,
        question: str,
        history: Optional[Iterable[Dict[str, str]]] = None,
    ) -> str:
        """Return a plain-language answer that always carries the disclaimer."""
        messages = build_messages(question, history)
        prompt = format_prompt(messages)
        raw = self._run_model(prompt) or ""
        answer = raw.split("<|im_end|>")[0].strip()
        return ensure_disclaimer(answer)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B plain-language health information chatbot"
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    args = parser.parse_args(argv)

    chatbot = SparkHealthChatbot()
    print(chatbot.generate(args.question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
