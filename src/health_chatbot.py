"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

Scope: plain-language *health education* only.
The model never diagnoses, never prescribes, and always points the reader to a
qualified clinician. The application code (not the model) is the safety gate.

Run:
    python src/health_chatbot.py                 # uses DEMO_QUESTION
    python src/health_chatbot.py "my question"   # ask your own question

Requires `transformers` + `torch` only when actually generating text; the
prompt-building helpers are dependency-free so tests run offline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

DISCLAIMER = (
    "This is general health information for education, not a diagnosis or a "
    "prescription. Please consult a qualified clinician about your own situation."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health service. "
    "Answer in plain, simple language that a person without medical training can "
    "understand, and keep the answer short (at most 200 words).\n"
    "Rules you must never break:\n"
    "1. Do not diagnose. Do not name a specific disease as the user's condition.\n"
    "2. Do not prescribe, recommend dosages, or suggest starting/stopping a medicine.\n"
    "3. Explain general concepts, common causes, and which kind of clinician or "
    "service usually helps.\n"
    "4. List clear 'seek urgent care now' warning signs when the topic could be "
    "serious.\n"
    "5. If the question is outside health education, say so briefly.\n"
    "6. End by telling the reader to consult a qualified clinician."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a cough and mild "
    "fever. What could be going on, and when should I see someone?"
)

_VALID_ROLES = ("user", "assistant")


def _clean_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Keep only well-formed user/assistant turns with non-empty text."""
    cleaned: List[Dict[str, str]] = []
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in _VALID_ROLES and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()})
    return cleaned


def build_messages(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the chat-message list sent to Spark-X2.6-4B.

    The system prompt always comes first and the user question always comes
    last, so the safety rules cannot be displaced by prior conversation.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_clean_history(history))
    messages.append({"role": "user", "content": question.strip()})
    return messages


def normalize_answer(text: str) -> str:
    """Trim model output and guarantee the safety disclaimer is present once."""
    answer = (text or "").strip()
    if not answer:
        answer = (
            "I could not produce a reliable answer. Please rephrase your question "
            "or ask a clinician directly."
        )
    if DISCLAIMER not in answer:
        answer = answer + "\n\n" + DISCLAIMER
    return answer


def load_model(model_id: str = MODEL_ID):
    """Load the released Spark-X2.6-4B weights and tokenizer (lazy import)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate(
    question: str,
    model=None,
    tokenizer=None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Answer one health-education question with Spark-X2.6-4B."""
    messages = build_messages(question, history)

    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=inputs.get("attention_mask").to(model.device)
        if inputs.get("attention_mask") is not None
        else None,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None),
    )
    text = tokenizer.decode(
        output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
    )
    return normalize_answer(text)


def main(argv: Optional[List[str]] = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION

    print(f"Model: {MODEL_ID}")
    print(f"Question: {question}\n")
    print(generate(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
