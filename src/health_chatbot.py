"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

Public-interest demo: plain-language health education for people who cannot
easily reach a clinician. The model explains and points to professional care;
it never diagnoses or prescribes.

Run the demo:

    python src/health_chatbot.py --demo
    python src/health_chatbot.py --question "I still have a cough after a trip"

Model: https://huggingface.co/XHToken/Spark-X2.6-4B
"""

from __future__ import annotations

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
MAX_HISTORY_TURNS = 6

EMERGENCY_NOTE = (
    "If this may be an emergency (chest pain, trouble breathing, heavy "
    "bleeding, fainting, or a very sick child), contact local emergency "
    "services right now."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information service. You explain health topics in simple, respectful "
    "language that a person with no medical training can understand.\n"
    "Rules you must follow:\n"
    "1. Never give a diagnosis and never prescribe or dose a medicine.\n"
    "2. Explain what is commonly true, which warning signs to watch for, and "
    "when to see a clinician.\n"
    "3. Always recommend confirming with a qualified local clinician or "
    "pharmacist.\n"
    "4. If the situation sounds urgent, tell the user to seek emergency care "
    "immediately.\n"
    "5. Do not ask for names, addresses, ID numbers or other identifying "
    "data.\n"
    "6. Answer in the language the user writes in. Keep answers short and use "
    "plain words."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a dry cough and "
    "some fever in the evening. I do not have much money and I am not sure if "
    "this is serious. What should I do?"
)

_STOP_MARKERS = (
    "<|end|>",
    "<|endoftext|>",
    "<|user|>",
    "<|system|>",
    "<|assistant|>",
)

_MODEL_CACHE = {}


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    ``history`` is an optional list of ``{"role": ..., "content": ...}`` turns
    from the current session. Only the last ``MAX_HISTORY_TURNS`` are kept.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    cleaned_history = []
    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("each history turn must be a dict")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant"):
            raise ValueError("history role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("history content must be a non-empty string")
        cleaned_history.append({"role": role, "content": content.strip()})

    if len(cleaned_history) > MAX_HISTORY_TURNS:
        cleaned_history = cleaned_history[-MAX_HISTORY_TURNS:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(cleaned_history)
    messages.append({"role": "user", "content": question.strip()})
    return messages


def format_prompt(messages, tokenizer=None):
    """Render messages as a prompt string.

    Uses the tokenizer chat template when one is available, otherwise falls
    back to a simple role-marked transcript.
    """
    if not messages:
        raise ValueError("messages must not be empty")

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    lines = []
    for message in messages:
        lines.append("<|%s|>\n%s" % (message["role"], message["content"]))
    lines.append("<|assistant|>\n")
    return "\n".join(lines)


def clean_reply(text):
    """Trim special tokens and whitespace from a generated reply."""
    if not isinstance(text, str):
        raise ValueError("reply must be a string")
    for marker in _STOP_MARKERS:
        index = text.find(marker)
        if index != -1:
            text = text[:index]
    return text.strip()


def generate_reply(question, history=None, chat_fn=None):
    """Return a dict with the model reply and the exact messages used.

    ``chat_fn`` may be injected for tests so that no weights are downloaded.
    """
    messages = build_messages(question, history)
    if chat_fn is None:
        chat_fn = _chat_with_spark
    reply = clean_reply(chat_fn(messages))
    if not reply:
        raise RuntimeError("model returned an empty reply")
    return {
        "model": MODEL_ID,
        "question": messages[-1]["content"],
        "reply": reply,
        "emergency_note": EMERGENCY_NOTE,
        "messages": messages,
    }


def _chat_with_spark(messages):
    """Load Spark-X2.6-4B once and generate a reply for ``messages``."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not _MODEL_CACHE:
        _MODEL_CACHE["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_ID)
        _MODEL_CACHE["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="auto"
        )

    tokenizer = _MODEL_CACHE["tokenizer"]
    model = _MODEL_CACHE["model"]

    prompt = format_prompt(messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health chatbot")
    parser.add_argument("--question", help="question to ask the assistant")
    parser.add_argument(
        "--demo", action="store_true", help="run the built-in demo question"
    )
    args = parser.parse_args(argv)

    question = args.question or DEMO_QUESTION
    result = generate_reply(question)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
