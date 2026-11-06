"""Minimal health-information chatbot built on Spark-X2.6-4B.

Public-interest demo for HER Hack-Astron #7 (Health + accessibility).
The assistant explains health topics in plain language, never diagnoses or
prescribes, and always points the user to a qualified clinician.

Run the dependency-free prompt demo:

    python src/health_chatbot.py

Actually load the model and generate a reply:

    python src/health_chatbot.py --run-model
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List, Optional, Tuple

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
MAX_QUESTION_CHARS = 2000
MAX_HISTORY_TURNS = 10

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public health-information service. "
    "Explain health topics in plain, simple language that a person without medical "
    "training can understand, and offer general, widely accepted self-care information. "
    "You must NOT diagnose, must NOT prescribe, and must NOT recommend specific "
    "prescription medicines or dosages. Never claim to replace a clinician. Always "
    "remind the user to consult a qualified health professional, and clearly list "
    "emergency warning signs (for example difficulty breathing, chest pain, fainting, "
    "confusion, or heavy bleeding) that mean the person should seek urgent care "
    "immediately. If the question is outside general health information, say so briefly "
    "and redirect. Never ask for or repeat personal identifying details. Keep answers "
    "short, structured, and non-alarming."
)

DEMO_QUESTION = (
    "Saya demam dan batuk sejak tiga hari lalu, dan baru pulang dari perjalanan. "
    "Apa yang sebaiknya saya perhatikan?"
)

EMERGENCY_NOTE = (
    "If you have trouble breathing, chest pain, fainting, confusion, or heavy bleeding, "
    "seek emergency care now. This information is educational only and does not replace "
    "advice from a qualified health professional."
)


def validate_question(question: str) -> str:
    """Trim and bound user input so prompts stay small and predictable."""
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must not be empty")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise ValueError(
            "question must be at most %d characters" % MAX_QUESTION_CHARS
        )
    return cleaned


def build_messages(
    question: str,
    history: Optional[Iterable[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B."""
    cleaned = validate_question(question)
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        recent = [
            turn
            for turn in history
            if isinstance(turn, dict)
            and turn.get("role") in ("user", "assistant")
            and turn.get("content")
        ][-MAX_HISTORY_TURNS:]
        for turn in recent:
            messages.append(
                {"role": turn["role"], "content": str(turn["content"]).strip()}
            )
    messages.append({"role": "user", "content": cleaned})
    return messages


def build_prompt_text(messages: List[Dict[str, str]]) -> str:
    """Fallback plain-text prompt used only when no chat template is available."""
    lines = []
    for message in messages:
        lines.append("%s: %s" % (message["role"], message["content"]))
    lines.append("assistant:")
    return "\n".join(lines)


def ensure_disclaimer(text: str) -> str:
    """Guarantee the emergency / not-medical-advice note is present."""
    body = (text or "").strip()
    if "emergency" in body.lower():
        return body or EMERGENCY_NOTE
    if not body:
        return EMERGENCY_NOTE
    return body + "\n\n" + EMERGENCY_NOTE


def load_model(model_id: str = MODEL_ID) -> Tuple[object, object]:
    """Load Spark-X2.6-4B lazily so the demo and tests need no heavy deps."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    return model, tokenizer


def generate_reply(
    question: str,
    history: Optional[Iterable[Dict[str, str]]] = None,
    model: object = None,
    tokenizer: object = None,
) -> str:
    """Generate a plain-language health-information reply with Spark-X2.6-4B."""
    messages = build_messages(question, history)
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = build_prompt_text(messages)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    reply = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
    return ensure_disclaimer(reply)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot (public-interest demo)"
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument(
        "--run-model",
        action="store_true",
        help="load XHToken/Spark-X2.6-4B and generate a reply",
    )
    args = parser.parse_args(argv)

    if not args.run_model:
        payload = {
            "model": MODEL_ID,
            "question": validate_question(args.question),
            "messages": build_messages(args.question),
            "note": "pass --run-model to generate a reply with Spark-X2.6-4B",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    reply = generate_reply(args.question)
    print(
        json.dumps(
            {"model": MODEL_ID, "question": args.question, "reply": reply},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
