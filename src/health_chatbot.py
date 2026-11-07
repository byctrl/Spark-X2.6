"""Minimal AI-for-Good health-information chatbot built on Spark-X2.6-4B.

The application answers general health-education questions in plain language.
It never diagnoses or prescribes: every reply carries a disclaimer and points
the reader to a qualified clinician. Emergency phrasing is detected in
application code so the notice does not depend on model output.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant inside a free public-interest "
    "health information service. Your reader may have limited literacy, "
    "limited internet access, or limited access to care.\n"
    "\n"
    "Rules:\n"
    "- Explain in short, plain sentences that a person without medical "
    "training can follow.\n"
    "- Do not diagnose, do not present a likely disease as a conclusion, and "
    "do not prescribe or recommend a specific medicine, dose, or treatment "
    "plan.\n"
    "- Always say what the person should do next and which kind of clinician "
    "or service can help.\n"
    "- If key facts are missing, list the missing facts instead of guessing.\n"
    "- If the question suggests an emergency, tell the person to seek "
    "emergency care immediately.\n"
    "- Answer in the language of the question.\n"
)

DISCLAIMER = (
    "Educational information only - not a diagnosis or a treatment plan. "
    "Please confirm any decision with a qualified clinician."
)

URGENT_NOTICE = (
    "This may be an emergency. Contact your local emergency number or go to "
    "the nearest emergency department now."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and a fever that comes back every evening. What could it be and what "
    "should I do next?"
)

URGENT_TERMS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "difficulty breathing",
    "severe bleeding",
    "heavy bleeding",
    "not breathing",
    "unconscious",
    "suicidal",
    "suicide",
    "stroke",
    "seizure",
    "anaphylaxis",
    "胸痛",
    "呼吸困难",
    "大出血",
    "昏迷",
    "自杀",
    "抽搐",
)


def needs_urgent_care(question: str) -> bool:
    """Return True when the question mentions an emergency warning phrase."""
    text = (question or "").lower()
    return any(term in text for term in URGENT_TERMS)


def build_messages(
    question: str,
    history: Optional[Sequence[Mapping[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt is always first and the new question is always last.
    The caller's ``history`` sequence is only read, never modified.
    """
    text = (question or "").strip()
    if not text:
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = (turn or {}).get("role")
        content = ((turn or {}).get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})
    return messages


def fallback_render(messages: Sequence[Mapping[str, str]]) -> str:
    """Render a prompt without relying on a tokenizer chat template."""
    labels = {"system": "System", "user": "User", "assistant": "Assistant"}
    lines = []
    for message in messages:
        label = labels.get(message.get("role", "user"), "User")
        lines.append("%s: %s" % (label, message.get("content", "")))
    lines.append("Assistant:")
    return "\n\n".join(lines)


def render_prompt(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    """Prefer the model chat template, fall back to a plain text render."""
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            return apply_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return fallback_render(messages)


def _prompt_length(inputs: Mapping[str, Any]) -> int:
    ids = inputs["input_ids"]
    shape = getattr(ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(ids)


def _move_to_device(inputs: Any, device: Any) -> Any:
    mover = getattr(inputs, "to", None)
    if callable(mover) and device is not None:
        return mover(device)
    return inputs


def load_model(model_id: str = MODEL_ID):
    """Load Spark-X2.6-4B and its tokenizer (imports transformers lazily)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    device_map = getattr(model, "device", None)
    if device_map is not None:
        try:
            model.to(device_map)
        except Exception:
            pass
    return model, tokenizer


def decorate_reply(reply: str, question: str) -> str:
    """Wrap a model answer with the safety notice and the disclaimer."""
    parts: List[str] = []
    if needs_urgent_care(question):
        parts.append(URGENT_NOTICE)
    cleaned = (reply or "").strip()
    if cleaned:
        parts.append(cleaned)
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def generate_reply(
    question: str,
    history: Optional[Sequence[Mapping[str, str]]] = None,
    model: Any = None,
    tokenizer: Any = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Run one Spark-X2.6-4B turn and return a safety-wrapped answer."""
    messages = build_messages(question, history)
    if tokenizer is None:
        if model is not None:
            raise ValueError("tokenizer is required when model is provided")
        model, tokenizer = load_model()
    elif model is None:
        raise ValueError("model is required when tokenizer is provided")

    prompt = render_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = _move_to_device(inputs, getattr(model, "device", None))
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    sequence = output[0]
    prompt_len = _prompt_length(inputs)
    reply = tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True)
    return decorate_reply(reply, question)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Health-information chatbot powered by Spark-X2.6-4B"
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    model, tokenizer = load_model(args.model_id)
    answer = generate_reply(
        args.question,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "model": args.model_id,
                    "question": args.question,
                    "answer": answer,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
