"""Minimal health-information chatbot built on Spark-X2.6-4B.

HER Hack-Astron #7 - Spark-X2.6 Open Source AI for Good Challenge.

The assistant is a patient-education helper: it explains health topics in
plain language, never diagnoses, never prescribes, and always points the
reader to a qualified clinician. Model generation is injected so the module
can be imported and tested without downloading weights.
"""

from __future__ import annotations

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in "
    "plain language that someone without medical training can understand. "
    "Never diagnose a condition, never name a likely disease as a conclusion, "
    "and never recommend, start, stop or change any medication or treatment. "
    "Point out clearly when symptoms may need urgent care. Always tell the "
    "reader to confirm anything important with a qualified clinician. When "
    "information is missing or uncertain, say so instead of guessing."
)

DISCLAIMER = (
    "This is general health information for education only, not medical "
    "advice. It cannot diagnose or treat a condition. Please confirm anything "
    "important with a qualified clinician, and seek urgent care if symptoms "
    "are severe or getting worse."
)

DEMO_QUESTION = (
    "I came back from a two-week trip ten days ago. For the last three days "
    "I have had a dry cough and a fever of about 38.2 C, and I feel more "
    "tired than usual. What should I know before I see a doctor?"
)

_VALID_HISTORY_ROLES = ("user", "assistant")


def build_messages(question, history=None):
    """Return OpenAI-style chat messages for one user question.

    ``history`` is an optional sequence of earlier ``{"role", "content"}``
    turns; it is copied, not mutated.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role not in _VALID_HISTORY_ROLES or not isinstance(content, str):
            raise ValueError(
                "history turns must look like "
                "{'role': 'user'|'assistant', 'content': str}"
            )
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def _default_generate(messages, max_new_tokens=MAX_NEW_TOKENS):
    """Run Spark-X2.6-4B locally with transformers (lazy import)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated = output[0][inputs.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def answer(question, generate=None, history=None, max_new_tokens=MAX_NEW_TOKENS):
    """Answer ``question`` and return the reply plus the safety disclaimer.

    ``generate`` is called as ``generate(messages, max_new_tokens=...)`` and
    must return a string. It defaults to local Spark-X2.6-4B inference.
    """
    generate = generate or _default_generate
    messages = build_messages(question, history)
    reply = generate(messages, max_new_tokens=max_new_tokens)

    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("model returned an empty reply")

    return {
        "question": question.strip(),
        "messages": messages,
        "reply": reply.strip(),
        "disclaimer": DISCLAIMER,
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a plain-language health question."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="question to ask; defaults to the built-in demo question",
    )
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip() or DEMO_QUESTION
    result = answer(question)

    print(result["reply"])
    print()
    print(result["disclaimer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
