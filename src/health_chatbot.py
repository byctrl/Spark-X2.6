"""Minimal health-information chatbot powered by Spark-X2.6-4B.

This module only *builds* prompts and formats them. The model call is done
lazily inside :func:`generate` so that importing this file never requires a
gPU, transformers, or network access.
"""

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who want plain-language "
    "health information.\n"
    "1. Use short, simple sentences and explain any medical word you must use.\n"
    "2. You do not diagnose and you do not treat. Never suggest prescription "
    "drugs, dosages, or a specific treatment plan, even when the user says "
    "they have a named condition.\n"
    "3. Always tell the user to confirm with a qualified clinician, nurse or "
    "pharmacist, and to ask which local service is right for them.\n"
    "4. If the description sounds urgent (chest pain, trouble breathing, "
    "fainting, heavy bleeding, signs of stroke, thoughts of self-harm), tell "
    "the user to contact emergency services right away.\n"
    "5. Never ask for or repeat identifying details such as full name, "
    "address, ID numbers or insurance numbers.\n"
    "6. Say clearly when you do not know, and point to a reliable source such "
    "as a national health service or a local clinic."
)

DEMO_QUESTION = (
    "I have had a dry cough and a mild fever since I got back from a trip two "
    "weeks ago. What should I do next?"
)

EMERGENCY_TERMS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "trouble breathing",
    "shortness of breath",
    "fainting",
    "passed out",
    "heavy bleeding",
    "stroke",
    "face drooping",
    "slurred speech",
    "self-harm",
    "suicide",
    "suicidal",
)


def build_messages(question, system_prompt=SYSTEM_PROMPT):
    """Return the chat messages for one user question.

    Raises ``ValueError`` when the question is empty or whitespace only.
    """
    if question is None or not question.strip():
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]


def format_prompt(messages):
    """Render chat messages as a ChatML-like prompt ending with an assistant turn."""
    lines = []
    for message in messages:
        lines.append("### {0}:".format(message["role"].capitalize()))
        lines.append(message["content"])
        lines.append("")
    lines.append("### Assistant:")
    return "\n".join(lines)


def is_urgent(text):
    """Return True when the text mentions a red-flag symptom."""
    if not text:
        return False
    lowered = text.lower()
    return any(term in lowered for term in EMERGENCY_TERMS)


def safety_notice(text):
    """Return the extra notice that should be shown above a reply."""
    if is_urgent(text):
        return (
            "This may be a medical emergency. Contact your local emergency "
            "number now, or ask someone nearby to call for you."
        )
    return (
        "This is general education only, not a diagnosis or treatment "
        "advice. Please confirm with a qualified clinician, nurse or "
        "pharmacist."
    )


def generate(question, max_new_tokens=MAX_NEW_TOKENS):
    """Run Spark-X2.6-4B on one question and return only the new text.

    Importing ``transformers`` happens here so that the rest of the module
    (and the unit tests) work without any model installed.
    """
    prompt = format_prompt(build_messages(question))
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "transformers is required for generate(); install it with "
            "`pip install transformers torch`."
        ) from exc

    pipe = pipeline("text-generation", model=MODEL_ID, device_map="auto")
    result = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    text = result[0]["generated_text"]
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def main(argv=None):  # pragma: no cover - simple CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Ask Spark-X2.6-4B a health question.")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    args = parser.parse_args(argv)

    print(safety_notice(args.question))
    print()
    print(generate(args.question))


if __name__ == "__main__":  # pragma: no cover
    main()
