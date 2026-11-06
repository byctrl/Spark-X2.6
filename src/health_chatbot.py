"""Minimal health-information chatbot demo for XHToken/Spark-X2.6-4B.

This is a deliberately small reference application for the HER Hack-Astron
AI-for-Good challenge: inclusive, plain-language patient education.

It is NOT a diagnostic tool. The assistant never gives a diagnosis or a
treatment plan; it explains things in simple language and points the user to a
qualified clinician. The application code keeps that boundary explicit so a
model mistake cannot silently turn into medical advice.

Run locally (requires `transformers` + the model weights):

    python src/health_chatbot.py
    python src/health_chatbot.py --question "..."
"""

from __future__ import annotations

import argparse

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in "
    "plain, simple language for a person with no medical training. "
    "You must NOT diagnose the user, must NOT name a probable disease, and "
    "must NOT recommend prescription medicines or dosages. "
    "Do not claim to replace a doctor. "
    "Answer in the user's language. "
    "Structure the answer as: (1) a short neutral explanation of the general "
    "topic, (2) which warning signs mean the person should seek care urgently, "
    "(3) practical next steps such as who to contact and what to prepare, and "
    "(4) a one-line reminder that a qualified clinician must confirm anything "
    "important. Keep it short and concrete."
)

DISCLAIMER = (
    "This is general education only, not medical advice or a diagnosis. "
    "Please have a qualified clinician review your situation."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago. Since then I have a dry cough and a "
    "fever that comes back in the evenings. I do not have a regular doctor yet "
    "and I am not sure whether this is serious. What should I know and what "
    "should I do next?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt is always first, so the safety boundary cannot be
    dropped by the caller. Invalid history turns are skipped instead of
    raising, but an empty question is a programming error.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def generate_answer(
    question,
    history=None,
    max_new_tokens=MAX_NEW_TOKENS,
    model=None,
    tokenizer=None,
):
    """Generate an educational answer with Spark-X2.6-4B.

    The model and tokenizer are loaded lazily so importing this module (and
    running the tests) does not need transformers or the model weights.
    Greedy decoding keeps the demo reproducible.
    """
    messages = build_messages(question, history)

    if tokenizer is None or model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEMO_QUESTION)
    args = parser.parse_args(argv)

    answer = generate_answer(args.question)
    print(answer)
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
