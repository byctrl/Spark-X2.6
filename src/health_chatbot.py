"""Minimal health-information chatbot built on Spark-X2.6-4B.

This is the smallest useful "AI for Good" demo for HER Hack-Astron #7:
a plain-language patient-education helper that never diagnoses or prescribes
and always points the user to a qualified clinician.

Run:
    python src/health_chatbot.py                 # uses DEMO_QUESTION
    python src/health_chatbot.py "my question"   # ask your own question

The model is loaded lazily so that importing this module (and running the
unit tests) does not require torch/transformers to be installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-health help desk. "
    "Explain health topics in plain language at roughly a 6th-grade reading level. "
    "You must NOT diagnose, prescribe, recommend doses, or replace a clinician. "
    "Never tell the user to stop or change a prescribed treatment. "
    "Use short paragraphs or a short bullet list. "
    "If red-flag symptoms are described (for example chest pain, trouble "
    "breathing, fainting, heavy bleeding, a fever in a newborn, or thoughts of "
    "self-harm), say clearly that this needs urgent in-person care or emergency "
    "services now. "
    "Otherwise, end every answer by suggesting the user confirm with a qualified "
    "health worker and by naming the kind of clinician or service that fits."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago and since yesterday I have "
    "a dry cough and a fever of 38.3 C. I also feel very tired. What should I do?"
)

SAFETY_REMINDER = (
    "This is general health education from a language model, not medical advice. "
    "Contact a qualified clinician or your local emergency number for care."
)


def build_messages(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat payload sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                raise ValueError(
                    "history turns need role in {'user','assistant'} and a string content"
                )
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question.strip()})
    return messages


_CACHE: Dict[str, Any] = {}


def load_model(model_id: str = MODEL_ID):
    """Load tokenizer and model once; transformers is imported lazily."""
    if "model" not in _CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        _CACHE["tokenizer"] = tokenizer
        _CACHE["model"] = model
    return _CACHE["tokenizer"], _CACHE["model"]


def generate_reply(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    model_id: str = MODEL_ID,
) -> str:
    """Generate one plain-language answer with Spark-X2.6-4B (greedy decoding)."""
    messages = build_messages(question, history)
    tokenizer, model = load_model(model_id)

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main(argv: Optional[List[str]] = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION

    print("Q:", question)
    print()
    print(generate_reply(question))
    print()
    print(SAFETY_REMINDER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
