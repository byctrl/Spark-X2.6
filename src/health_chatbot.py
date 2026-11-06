"""Minimal health-information chatbot built on Spark-X2.6-4B (issue #16 demo).

The chatbot is a patient-education helper: it explains health topics in plain
language, never diagnoses or prescribes, and points people to a clinician.
Heavy dependencies (transformers/torch) are imported lazily inside
``HealthChatbot._load`` so the helper functions and the tests stay usable on a
machine without the model or GPU.
"""

from __future__ import annotations

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who need clear health "
    "information.\n"
    "Follow these rules:\n"
    "1. use simple, everyday language and explain any medical word you must use.\n"
    "2. never provide a diagnosis or a treatment plan. do not tell the user which\n"
    "   disease they have, and do not tell them which medicine to take.\n"
    "3. encourage the user to see a qualified clinician or a local health service.\n"
    "   say clearly when a symptom needs urgent care, for example trouble breathing,\n"
    "   chest pain, fainting, heavy bleeding, or a fever that will not go away.\n"
    "4. do not ask for or repeat identifying details such as names, addresses, or id\n"
    "   numbers. do not claim to be a doctor.\n"
    "5. if the question is outside health education, say so briefly and suggest a\n"
    "   better source.\n"
    "6. answer in the language the user writes in."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days, since I came back from a "
    "trip. What should I know before I see a doctor?"
)

SAFETY_NOTE = (
    "This is general health education, not a diagnosis. Please see a qualified "
    "clinician or a local health service for advice about your own situation."
)

URGENT_KEYWORDS = (
    "chest pain",
    "trouble breathing",
    "difficulty breathing",
    "cannot breathe",
    "shortness of breath",
    "fainting",
    "unconscious",
    "heavy bleeding",
    "seizure",
    "stroke",
    "suicidal",
)

_CLINICIAN_WORDS = (
    "clinician",
    "doctor",
    "nurse",
    "health worker",
    "health service",
    "medical professional",
)


def build_messages(question, history=None):
    """Return chat messages: one system prompt, optional history, user turn."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": (question or "").strip()})
    return messages


def format_prompt(tokenizer, messages):
    """Render messages with the tokenizer chat template, or a plain fallback."""
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_template is not None:
        return apply_template(messages, tokenize=False, add_generation_prompt=True)
    lines = ["%s: %s" % (m["role"].upper(), m["content"]) for m in messages]
    return "\n\n".join(lines) + "\n\nASSISTANT:"


def has_urgent_symptom(text):
    """True when the user text mentions a red-flag symptom."""
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in URGENT_KEYWORDS)


def strip_prompt(text, prompt):
    """Drop an echoed prompt when the runtime returns the full text."""
    text = (text or "").strip()
    prompt = (prompt or "").strip()
    if prompt and text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text


def clean_response(text):
    """Trim whitespace and stray chat role markers from a model answer."""
    text = (text or "").strip()
    for marker in ("USER:", "ASSISTANT:", "SYSTEM:"):
        if text.endswith(marker):
            text = text[: -len(marker)].strip()
    return text


def ensure_safety_note(answer):
    """Append the clinician reminder when the answer omits it."""
    text = (answer or "").strip()
    lowered = text.lower()
    if any(word in lowered for word in _CLINICIAN_WORDS):
        return text
    if not text:
        return SAFETY_NOTE
    return text + "\n\n" + SAFETY_NOTE


def finalize_answer(text):
    """Turn raw model text into a user-facing answer."""
    return ensure_safety_note(clean_response(text))


class HealthChatbot:
    """Thin wrapper around a local Spark-X2.6-4B text-generation pipeline."""

    def __init__(self, model_id=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._generator = None

    def _load(self):
        if self._generator is None:
            from transformers import pipeline  # imported lazily on purpose

            self._generator = pipeline(
                "text-generation",
                model=self.model_id,
                device_map="auto",
            )
        return self._generator

    def reply(self, question, history=None):
        messages = build_messages(question, history)
        generator = self._load()
        prompt = format_prompt(getattr(generator, "tokenizer", None), messages)
        outputs = generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        if isinstance(outputs, list) and outputs:
            raw = outputs[0].get("generated_text", "")
        else:
            raw = str(outputs)
        return finalize_answer(strip_prompt(raw, prompt))


def answer_question(question, chatbot=None):
    """Convenience helper used by the demo entry point."""
    bot = chatbot or HealthChatbot()
    return bot.reply(question)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a plain-language health question."
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args(argv)

    bot = HealthChatbot(model_id=args.model)
    print(bot.reply(args.question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
