"""Minimal Spark-X2.6-4B health-knowledge chatbot (AI for Good demo).

Patient-education assistant: explains health information in plain language,
refuses diagnosis/prescription, and always points the user to a qualified
health worker. Built for people with limited health literacy or limited
access to care.

Usage:
    python src/health_chatbot.py                 # run the demo question
    python src/health_chatbot.py "my question"    # ask your own question

The model is loaded lazily; importing this module never downloads weights.
"""

from __future__ import annotations

import argparse
import sys

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who may have limited "
    "health literacy or limited access to care. "
    "Explain health information in simple, respectful language (around a 6th "
    "grade reading level) and define any medical term you use. "
    "You must NOT diagnose, prescribe, suggest dosages, or tell the user to "
    "start, stop or change a treatment. Do not claim to be a clinician. "
    "If information is missing, ask one short clarifying question instead of "
    "guessing, and say clearly which facts are general knowledge rather than "
    "advice about this person. "
    "Always end by advising the user to consult a qualified health worker and "
    "name the kind of clinician or service that fits the question. "
    "If the question suggests an emergency (for example chest pain, trouble "
    "breathing, heavy bleeding, fainting, or thoughts of self-harm), say "
    "first and clearly that the person should seek emergency care now. "
    "Reply in the language the user wrote in."
)

EMERGENCY_KEYWORDS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "trouble breathing",
    "shortness of breath",
    "heavy bleeding",
    "severe bleeding",
    "passed out",
    "fainting",
    "unconscious",
    "suicide",
    "kill myself",
    "self-harm",
    "seizure",
    "stroke",
)

EMERGENCY_NOTE = (
    "This may be an emergency. Please contact your local emergency number or "
    "go to the nearest emergency department now."
)

EMPTY_REPLY_NOTE = (
    "The model returned an empty answer. Please try rephrasing the question, "
    "and talk to a qualified health worker if the concern continues."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a cough and "
    "fever in the evenings. I do not have insurance. What should I know before "
    "I see someone, and what kind of clinic should I look for?"
)


def detect_emergency(text: str) -> bool:
    """Cheap keyword screen so urgent wording never waits for model output."""

    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


def build_messages(question: str, extra_context: str | None = None):
    """Build the chat messages sent to Spark-X2.6-4B."""

    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    context = (extra_context or "").strip()
    if context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Reference material supplied by the user. Use it only if it "
                    "is relevant, and do not treat it as a diagnosis:\n" + context
                ),
            }
        )

    messages.append({"role": "user", "content": question})
    return messages


def build_prompt(messages, tokenizer=None) -> str:
    """Render messages with the tokenizer chat template, or a plain fallback."""

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    parts = [
        "<|im_start|>{role}\n{content}<|im_end|>".format(
            role=message["role"], content=message["content"]
        )
        for message in messages
    ]
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def answer_question(
    question: str,
    generator,
    extra_context: str | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """Run one turn. ``generator(prompt, max_new_tokens) -> str`` is injected."""

    messages = build_messages(question, extra_context)
    prompt = build_prompt(messages)
    reply = (generator(prompt, max_new_tokens) or "").strip()

    emergency = detect_emergency(question)
    if not reply:
        reply = EMPTY_REPLY_NOTE
    if emergency and EMERGENCY_NOTE not in reply:
        reply = EMERGENCY_NOTE + "\n\n" + reply

    return {
        "model": MODEL_ID,
        "question": question,
        "emergency": emergency,
        "answer": reply,
    }


class SparkGenerator:
    """Lazy transformers wrapper around Spark-X2.6-4B."""

    def __init__(self, model_id: str = MODEL_ID, device_map: str = "auto"):
        self.model_id = model_id
        self.device_map = device_map
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map=self.device_map
        )

    def __call__(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        self._load()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health chatbot demo")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--context", default=None, help="optional reference text")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    result = answer_question(
        args.question,
        SparkGenerator(model_id=args.model),
        extra_context=args.context,
        max_new_tokens=args.max_new_tokens,
    )

    print("Model: " + result["model"])
    print("\nQuestion: " + result["question"])
    print("\nAnswer:\n" + result["answer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
