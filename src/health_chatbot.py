"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

The chatbot answers general health-education questions in plain language.
It never diagnoses, prescribes, or replaces a qualified clinician: the
system prompt forbids that, and every reply is followed by a disclaimer.

Run the demo (requires a local Spark-X2.6 runtime):

    python src/health_chatbot.py "I have had a cough and fever for days"

No question is passed -> the DEMO_QUESTION scenario is used.
"""

import sys

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

DISCLAIMER = (
    "This is general health information for education only. It is not a "
    "diagnosis or a treatment plan. Please contact a qualified health "
    "worker or your local emergency service if your symptoms are severe "
    "or getting worse."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "service. Use simple, plain language and short sentences that are "
    "easy to read on a phone. Explain common health topics, warning "
    "signs, and when a person should seek care. Do NOT give a diagnosis, "
    "do NOT recommend prescription medicines or doses, and do NOT "
    "replace a clinician. If the question needs a diagnosis or urgent "
    "care, say so clearly and direct the person to a qualified health "
    "worker or emergency service."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days since I came "
    "back from a trip. What should I do?"
)


def build_messages(question):
    """Build the chat messages sent to the model for one user question."""
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    text = question.strip()
    if not text:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def render_prompt(messages):
    """Render messages as a plain prompt.

    This is a portable fallback so the demo works without a tokenizer.
    Production deployments should prefer ``tokenizer.apply_chat_template``.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    parts = []
    for message in messages:
        parts.append(
            "<|im_start|>{role}\n{content}<|im_end|>".format(
                role=message["role"], content=message["content"]
            )
        )
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def finalize_reply(text):
    """Clean the model output and guarantee the safety disclaimer."""
    if not isinstance(text, str):
        raise ValueError("model output must be a string")
    cleaned = text.strip()
    if not cleaned:
        return (
            "I could not produce a safe answer for this question. Please "
            "ask a qualified health worker.\n\n" + DISCLAIMER
        )
    if DISCLAIMER in cleaned:
        return cleaned
    return cleaned + "\n\n" + DISCLAIMER


def load_generator(model_id=MODEL_ID):
    """Load Spark-X2.6 locally and return a prompt -> text callable.

    ``transformers`` is imported lazily so the module and its tests stay
    dependency-free.
    """
    from transformers import pipeline

    pipe = pipeline("text-generation", model=model_id, device_map="auto")

    def _generate(prompt, max_new_tokens=MAX_NEW_TOKENS):
        output = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return output[0]["generated_text"]

    return _generate


def generate_reply(question, generator=None):
    """Answer one health-education question with Spark-X2.6-4B."""
    prompt = render_prompt(build_messages(question))
    if generator is None:
        generator = load_generator()
    raw = generator(prompt, MAX_NEW_TOKENS)
    return finalize_reply(raw)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print("Model:", MODEL_ID)
    print("Question:", question)
    try:
        print()
        print(generate_reply(question))
    except ImportError as exc:
        print("Spark-X2.6 runtime is not installed:", exc)
        print("Install it with: pip install transformers torch accelerate")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
