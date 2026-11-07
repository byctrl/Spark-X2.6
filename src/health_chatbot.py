"""Minimal Spark-X2.6-4B health-literacy chatbot demo.

Application for the Spark-X2.6 "AI for Good" challenge: an inclusive,
plain-language health knowledge assistant that runs on Spark-X2.6-4B.

Safety boundaries are enforced in application code, not left to the model:

* ``needs_emergency_escalation`` scans the user's text for red flags and, if
  one is found, returns a fixed escalation message without calling the model.
* ``SYSTEM_PROMPT`` forbids diagnosis and prescription and requires a plain
  language closing disclaimer on every answer.
* The reply is general patient education only; a clinician stays the
  decision maker.

The heavy ML dependencies (``torch``, ``transformers``) are imported lazily
inside ``load_model`` so importing this module never downloads weights.

Usage::

    python src/health_chatbot.py
    python src/health_chatbot.py "Is a mild fever after a trip a concern?"
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

DISCLAIMER = (
    "This is general information, not medical advice - please check with a "
    "qualified health worker about your own situation."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant. You explain general health "
    "information in plain, respectful language for people with low health "
    "literacy, including readers who use a screen reader.\n"
    "Rules you must always follow:\n"
    "1. Never diagnose. Do not tell the user what disease they have and do "
    "not prescribe medicines, doses or treatments.\n"
    "2. Explain what the described signs generally mean, what a clinician "
    "may want to check, and which questions the user can ask at an "
    "appointment.\n"
    "3. Use short sentences and everyday words. Expand any medical term you "
    "must use. Keep paragraphs to four sentences or fewer.\n"
    "4. Say clearly that the information is general and may not fit the "
    "user's own situation.\n"
    "5. If the description suggests an emergency - for example chest pain, "
    "difficulty breathing, heavy bleeding, fainting, new confusion, a fever "
    "in a newborn, or thoughts of self-harm - tell the user to contact local "
    "emergency services or a crisis line immediately.\n"
    "6. Never ask for or repeat names, addresses, ID numbers or other "
    "identifying details.\n"
    "7. Close every answer with this exact sentence: \"" + DISCLAIMER + "\""
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days, and I got back from "
    "a trip two weeks ago. What should I know before I see a doctor?"
)

EMERGENCY_PATTERNS: Sequence[str] = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "difficulty breathing",
    "trouble breathing",
    "shortness of breath",
    "heavy bleeding",
    "bleeding a lot",
    "unconscious",
    "fainted",
    "fainting",
    "seizure",
    "suicide",
    "suicidal",
    "kill myself",
    "self harm",
    "self-harm",
    "harm myself",
)


def needs_emergency_escalation(text: str) -> bool:
    """Return True when the user's text contains an emergency red flag."""
    if not text:
        return False
    lowered = text.lower()
    return any(pattern in lowered for pattern in EMERGENCY_PATTERNS)


def emergency_message() -> str:
    """Fixed, model-free escalation answer for red-flag descriptions."""
    return (
        "What you describe may be an emergency. Please contact your local "
        "emergency number or go to the nearest emergency department now, and "
        "do not wait for an online answer. If you are thinking about harming "
        "yourself, please call a local crisis line or emergency service "
        "immediately and try to stay with someone you trust.\n\n" + DISCLAIMER
    )


def build_messages(
    question: str,
    history: Optional[Sequence[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B.

    ``history`` is an optional sequence of earlier ``{"role", "content"}``
    turns. Only "user" and "assistant" roles are accepted so that callers
    cannot silently replace the system prompt.
    """
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": cleaned})
    return messages


def _chatml_fallback(messages: Sequence[Dict[str, str]]) -> str:
    """Generic ChatML rendering used when the tokenizer has no chat template."""
    parts = []
    for message in messages:
        parts.append(
            "<|im_start|>" + message["role"] + "\n" + message["content"] + "<|im_end|>"
        )
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def build_prompt(
    tokenizer,
    question: str,
    history: Optional[Sequence[Dict[str, str]]] = None,
) -> str:
    """Render the conversation to a single prompt string."""
    messages = build_messages(question, history)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            return apply_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return _chatml_fallback(messages)


def load_model(model_id: str = MODEL_ID):
    """Load the tokenizer and model. Downloads weights on first use."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def generate_reply(
    question: str,
    tokenizer=None,
    model=None,
    history: Optional[Sequence[Dict[str, str]]] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Return a plain-language answer, or an escalation for red flags."""
    if needs_emergency_escalation(question):
        return emergency_message()
    if tokenizer is None or model is None:
        raise ValueError("tokenizer and model are required for non-emergency questions")

    prompt = build_prompt(tokenizer, question, history)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[-1]
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output[0][prompt_length:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if DISCLAIMER not in reply:
        reply = reply + "\n\n" + DISCLAIMER
    return reply


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    print("Model:", MODEL_ID)
    print("Question:", question)
    print()
    if needs_emergency_escalation(question):
        print(emergency_message())
        return 0
    tokenizer, model = load_model(MODEL_ID)
    print(generate_reply(question, tokenizer, model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
