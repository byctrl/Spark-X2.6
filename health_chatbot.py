"""Minimal health-information chatbot powered by Spark-X2.6-4B.

Scope: patient *education* only. The assistant explains common health
concepts in plain language, never diagnoses, never suggests medicines or
dosages, and always redirects the reader to a qualified clinician.

This is the demo application for HER Hack-Astron #7 (AI for Good):
make health knowledge easier to reach for people who cannot easily
navigate dense medical text.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health topics in simple, "
    "respectful language that a person without medical training can follow. "
    "You must not diagnose, must not prescribe or name treatments or dosages, "
    "and must not tell the reader to stop or change a prescribed medicine. "
    "State clearly that your answer is general information, not medical advice, "
    "and tell the reader to consult a qualified clinician for their own "
    "situation. If the question suggests an emergency, advise contacting local "
    "emergency services immediately."
)

DISCLAIMER = (
    "General health information only - not a diagnosis or a prescription. "
    "Please consult a qualified clinician about your own situation."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak tiga hari lalu, dan baru pulang "
    "dari perjalanan. Hal apa yang sebaiknya saya perhatikan?"
)

GenerateFn = Callable[[List[Dict[str, str]], int], str]


def build_messages(question: str) -> List[Dict[str, str]]:
    """Wrap a user question into the Spark chat message list."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def format_prompt(messages: List[Dict[str, str]]) -> str:
    """ChatML fallback prompt used when the tokenizer has no chat template."""
    parts = [
        "<|im_start|>{role}\n{content}<|im_end|>\n".format(
            role=message["role"], content=message["content"]
        )
        for message in messages
    ]
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def render_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """Prefer the tokenizer chat template, fall back to ChatML."""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return format_prompt(messages)


def load_generate_fn(model_id: str = MODEL_ID) -> GenerateFn:
    """Load Spark-X2.6-4B locally and return a generate function.

    Imported lazily so that the tests and the prompt logic do not need the
    model weights or a GPU.
    """
    from transformers import pipeline  # local import: heavy dependency

    pipe = pipeline("text-generation", model=model_id, tokenizer=model_id)

    def generate_fn(messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        result = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
        generated = result[0]["generated_text"]
        if isinstance(generated, list):
            return str(generated[-1]["content"])
        return str(generated)

    return generate_fn


def answer(
    question: str,
    generate_fn: GenerateFn | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Return a plain-language explanation plus the safety disclaimer."""
    messages = build_messages(question)
    if generate_fn is None:
        generate_fn = load_generate_fn()
    text = (generate_fn(messages, max_new_tokens) or "").strip()
    if not text:
        raise RuntimeError("model returned an empty response")
    return text + "\n\n" + DISCLAIMER


def main() -> None:
    print("Model:", MODEL_ID)
    print("Question:", DEMO_QUESTION)
    print()
    print(answer(DEMO_QUESTION))


if __name__ == "__main__":
    main()
