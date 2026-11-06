"""Minimal health-information chatbot built on Spark-X2.6-4B.

Public-interest demo for HER Hack-Astron #7 (AI for Good). The model is used
for patient education only: it explains health topics in plain language, flags
warning signs, and refers the user to a qualified clinician. It never
diagnoses and never prescribes.

Model: https://huggingface.co/XHToken/Spark-X2.6-4B

Usage::

    python health_chatbot.py                 # runs the demo question
    python health_chatbot.py "my question"   # answers a custom question

This module imports only the Python standard library. ``transformers`` and
``torch`` are imported lazily inside :func:`load_model`, so the helper
functions and the unit tests run without a GPU or a model download.
"""

from __future__ import annotations

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant in a public-interest health "
    "information service. Answer in the user's language, in short sentences "
    "and simple everyday words.\n"
    "Rules you always follow:\n"
    "1. Explain health concepts; never diagnose the user and never state that "
    "a named disease is the cause of their symptoms.\n"
    "2. Never prescribe medicines, doses, or treatments.\n"
    "3. Point out warning signs that need urgent in-person care.\n"
    "4. Recommend a qualified clinician or the local emergency service.\n"
    "5. If the question is outside health education, say so briefly and offer "
    "what you can help with.\n"
    "6. Do not ask for or repeat identifying details such as full name, "
    "address, or ID numbers."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam 38 derajat sejak tiga hari lalu, setelah "
    "pulang dari perjalanan. Apa yang sebaiknya saya perhatikan?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt always comes first so the safety rules travel with
    every request; optional earlier turns are kept in order.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": str(question).strip()})
    return messages


def fallback_prompt(messages):
    """Plain role-tagged prompt used when no chat template is available."""
    parts = []
    for message in messages:
        parts.append("<|{}|>\n{}\n".format(message["role"], message["content"]))
    parts.append("<|assistant|>\n")
    return "".join(parts)


def build_prompt(tokenizer, messages):
    """Render messages with the tokenizer chat template, with a safe fallback."""
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            return apply_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return fallback_prompt(messages)


def generate_reply(
    question,
    history=None,
    model=None,
    tokenizer=None,
    max_new_tokens=MAX_NEW_TOKENS,
):
    """Return the assistant text for ``question``.

    ``model`` and ``tokenizer`` may be injected (tests, notebooks); when they
    are omitted the model is loaded from the Hugging Face hub once.
    """
    messages = build_messages(question, history)
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    prompt = build_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    prompt_length = inputs["input_ids"].shape[-1]
    new_tokens = output[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_model(model_id=MODEL_ID):
    """Load Spark-X2.6-4B and its tokenizer (requires transformers + torch)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    model.eval()
    return model, tokenizer


def main(argv=None):
    """Small CLI demo: answer one question, defaulting to the demo scenario."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    print("Model: {}".format(MODEL_ID))
    print("Q: {}".format(question))
    print()
    print(generate_reply(question))
    print()
    print("Note: educational information only, not a diagnosis. "
          "Please see a qualified clinician for personal advice.")


if __name__ == "__main__":
    main()
