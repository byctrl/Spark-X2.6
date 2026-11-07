"""Minimal health-information chatbot demo powered by Spark-X2.6-4B.

Public-interest demo (HER Hack-Astron #7): helps people read general health
information in plain language. It does not diagnose, does not prescribe, and
does not replace a qualified clinician.

Run:
    python src/health_chatbot.py "my question"
"""

from __future__ import annotations

import argparse

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain general health information "
    "in plain, simple language that a non-specialist can understand. Do not "
    "diagnose the person, do not recommend specific medicines or doses, and do "
    "not claim to replace a clinician. If the question suggests an emergency or "
    "a serious symptom, tell the person to seek urgent professional care. End "
    "with a short reminder to consult a qualified clinician."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and now I have a cough and a fever. "
    "What should I know before I see a doctor?"
)


def build_messages(question, history=None):
    """Build a chat-style message list: system prompt, history, then user turn."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def format_prompt(messages):
    """Render a message list into a plain text prompt (fallback for runtimes
    without a chat template)."""
    parts = []
    for message in messages:
        parts.append("<|%s|>\n%s\n" % (message["role"], message["content"]))
    parts.append("<|assistant|>\n")
    return "".join(parts)


def generate(question, history=None, model=None, tokenizer=None):
    """Return the model's answer as text.

    ``model`` and ``tokenizer`` may be injected (useful for testing or for
    reusing a locally loaded Spark-X2.6-4B). When omitted, the model is loaded
    lazily so importing this module does not require transformers.
    """
    messages = build_messages(question, history)

    if model is None or tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B plain-language health information demo"
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    args = parser.parse_args(argv)
    print(generate(args.question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
