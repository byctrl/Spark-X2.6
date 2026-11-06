"""Minimal Spark-X2.6-4B chatbot for inclusive health information.

Public-interest demo for the HER Hack-Astron #7 "AI for Good" challenge.
The assistant explains general health topics in plain language, always points
the user to a qualified clinician, and never diagnoses or prescribes.

Run locally with Spark-X2.6-4B available through ``transformers``::

    python src/health_chatbot.py

The model is loaded lazily, so importing this module (for example in tests)
does not download any weights.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information service. Explain general health topics in simple, respectful "
    "language that a person without medical training can follow.\n"
    "Rules you must always follow:\n"
    "1. Do not diagnose, do not name a disease as the user's own condition, and "
    "do not prescribe medicines, doses, or treatments.\n"
    "2. Do not claim to replace a doctor, nurse, pharmacist, or any other "
    "clinician; always encourage the user to speak with a qualified clinician.\n"
    "3. Tell the user which symptoms mean they should seek care urgently.\n"
    "4. Keep the answer short and structured, with at most a few bullet points.\n"
    "5. If the question is unclear, ask one short clarifying question."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for three days, and I travelled "
    "abroad last week. What should I do?"
)

Messages = List[Dict[str, str]]


def build_messages(question: str, history: Optional[Messages] = None) -> Messages:
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt always comes first, then any earlier conversation turns,
    then the current user question (stripped). An empty question is rejected so
    the model is never called with a blank prompt.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    messages: Messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    return messages


def render_prompt(messages: Messages) -> str:
    """Render chat messages into a plain-text prompt (chat-template fallback)."""
    lines = [f"{message['role'].upper()}: {message['content']}" for message in messages]
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def build_prompt(
    question: str,
    history: Optional[Messages] = None,
    tokenizer=None,
) -> str:
    """Return the prompt string for ``question``.

    When a tokenizer with a chat template is supplied it is used; otherwise a
    simple role-labelled prompt is rendered so the demo also works offline.
    """
    messages = build_messages(question, history=history)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return render_prompt(messages)


_generator = None


def load_generator(model_id: str = MODEL_ID):
    """Load (once) the local Spark-X2.6-4B text-generation pipeline."""
    global _generator
    if _generator is None:
        from transformers import pipeline

        _generator = pipeline("text-generation", model=model_id, device_map="auto")
    return _generator


def _generate_with_spark(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Default generator: run the local Spark-X2.6-4B pipeline."""
    pipe = load_generator()
    result = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    if isinstance(result, list) and result:
        item = result[0]
        if isinstance(item, dict):
            generated = item.get("generated_text", "")
            if isinstance(generated, list) and generated:
                return str(generated[-1].get("content", ""))
            return str(generated)
        return str(item)
    return str(result)


def answer(
    question: str,
    history: Optional[Messages] = None,
    generate: Optional[Callable[[str, int], str]] = None,
) -> str:
    """Answer a health question with Spark-X2.6-4B.

    ``generate`` is an optional ``(prompt, max_new_tokens) -> str`` callable used
    by tests and offline demos; when omitted the local Spark pipeline is used.
    """
    prompt = build_prompt(question, history=history)
    generator = generate if generate is not None else _generate_with_spark
    return (generator(prompt, MAX_NEW_TOKENS) or "").strip()


def chat(
    question: str,
    history: Optional[Messages] = None,
    generate: Optional[Callable[[str, int], str]] = None,
) -> str:
    """Alias for :func:`answer`, kept for readability in demo code."""
    return answer(question, history=history, generate=generate)


def main() -> int:
    print(f"Model: {MODEL_ID}")
    print(f"Demo question: {DEMO_QUESTION}\n")
    print(answer(DEMO_QUESTION))
    print("\nThis assistant gives general information only and does not replace a clinician.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
