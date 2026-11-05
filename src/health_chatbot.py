"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Scope: plain-language patient education only. The assistant must never
diagnose or prescribe, and always routes personal decisions to a qualified
clinician. Heavy ML dependencies (torch / transformers) are imported lazily
so that prompt-building helpers and tests stay importable on a bare Python.
"""

from __future__ import annotations

from typing import Any, Dict, List

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information service. Explain health topics in plain, respectful "
    "language that a non-specialist can understand. Never diagnose a "
    "condition, never prescribe medicine or dosages, and never claim to "
    "replace a clinician. If the question suggests an emergency, tell the "
    "user to contact local emergency services immediately. Always end by "
    "advising the user to consult a qualified health professional before "
    "making personal medical decisions."
)

DEMO_QUESTION = (
    "Saya baru pulang dari perjalanan dan sekarang batuk kering serta demam "
    "ringan sejak dua hari lalu. Apa yang sebaiknya saya perhatikan?"
)

SAFETY_NOTE = (
    "This demo provides general health education only. It is not a diagnosis "
    "and not a substitute for care from a qualified health professional."
)


def build_messages(
    question: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """Build the chat messages sent to Spark-X2.6-4B.

    Raises:
        ValueError: if ``question`` is not a non-empty string.
    """
    if not isinstance(question, str):
        raise ValueError("question must be a non-empty string")
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must be a non-empty string")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": cleaned},
    ]


def build_prompt(messages: List[Dict[str, str]], tokenizer: Any) -> str:
    """Render chat messages into a single prompt string.

    Uses the tokenizer's chat template when available and falls back to a
    simple role-prefixed transcript otherwise.
    """
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            rendered = apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except (TypeError, ValueError):
            pass
    lines = [f"{message['role']}: {message['content']}" for message in messages]
    lines.append("assistant:")
    return "\n".join(lines)


def load_model(model_id: str = MODEL_ID) -> Any:
    """Load the Spark-X2.6-4B tokenizer and model (lazy heavy imports).

    Returns:
        A ``(model, tokenizer)`` tuple.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "transformers is required to run the model. Install it with "
            "`pip install transformers torch`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_reply(
    question: str,
    model: Any = None,
    tokenizer: Any = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate a plain-language health-education reply for ``question``.

    The caller remains responsible for showing :data:`SAFETY_NOTE` next to the
    reply; model output is never treated as medical ground truth.
    """
    messages = build_messages(question)
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    prompt = build_prompt(messages, tokenizer)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "torch is required to run the model. Install it with "
            "`pip install torch`."
        ) from exc

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
        )
    generated = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> int:  # pragma: no cover - manual demo entry point
    print(f"Model: {MODEL_ID}")
    print(f"Question: {DEMO_QUESTION}")
    print(f"\nReply:\n{generate_reply(DEMO_QUESTION)}")
    print(f"\n{SAFETY_NOTE}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
