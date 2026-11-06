"""Minimal health-information chatbot powered by Spark-X2.6-4B.

Demo entry for the Spark-X2.6 Open Source AI for Good Challenge
(health / accessibility track). The model is used as a plain-language
patient-education assistant: it explains general health topics, but never
diagnoses, prescribes, or replaces a qualified clinician.

Run the demo with::

    python src/health_chatbot.py

``transformers`` is imported lazily inside :func:`_load_pipeline`, so this
module and its tests can be imported without downloading any model weights.
"""

from __future__ import annotations

import argparse
import sys

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain general health topics in "
    "simple, respectful language that a non-specialist can understand, using "
    "short paragraphs or bullet points.\n"
    "Rules you must always follow:\n"
    "1. Do not diagnose and do not name a likely disease as a conclusion. Do "
    "not prescribe medicines, doses, home remedies or treatments.\n"
    "2. Explain what is generally known about the topic, which warning signs "
    "need urgent care, and which questions the person can ask a clinician.\n"
    "3. Always tell the user to consult a qualified health professional about "
    "their own situation, and to contact emergency services for severe or "
    "rapidly worsening symptoms.\n"
    "4. If the request is outside general health education, say so briefly and "
    "point to a better source.\n"
    "5. Never ask for or repeat identifying details such as full name, "
    "address, ID number or insurance number."
)

DEMO_QUESTION = (
    "I came back from a two-week trip ten days ago. Since yesterday I have a "
    "dry cough and a fever of 38.2 C. What should I pay attention to, and when "
    "should I see a doctor?"
)

DISCLAIMER = (
    "This is general health information, not medical advice. "
    "Please consult a qualified clinician about your own situation."
)


def build_messages(question, history=None):
    """Build the chat messages sent to the model.

    ``history`` is an optional list of previous ``{"role", "content"}`` turns
    (roles ``user`` / ``assistant``) so a caller can keep a short conversation.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("history turns must be dicts")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError(
                "history turns need a role of user/assistant and string content"
            )
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def _load_pipeline(model_id=MODEL_ID):
    """Load Spark-X2.6-4B as a text-generation chat pipeline (lazy import)."""
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "transformers is required to run the model: "
            "pip install transformers accelerate"
        ) from exc

    return pipeline("text-generation", model=model_id, device_map="auto")


def _extract_reply(output):
    """Normalize a pipeline result into a plain reply string."""
    first = output[0] if isinstance(output, (list, tuple)) and output else output
    if isinstance(first, dict):
        generated = first.get("generated_text", "")
    else:
        generated = first

    if isinstance(generated, (list, tuple)) and generated:
        last = generated[-1]
        if isinstance(last, dict):
            return str(last.get("content", "")).strip()
        return str(last).strip()

    return str(generated or "").strip()


def generate(question, history=None, generator=None, max_new_tokens=MAX_NEW_TOKENS):
    """Return a plain-language answer for ``question``.

    ``generator`` accepts any callable with the ``transformers``
    text-generation chat-pipeline signature. It is injected by the tests so
    they never download model weights.
    """
    messages = build_messages(question, history)
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")

    if generator is None:
        generator = _load_pipeline()

    output = generator(messages, max_new_tokens=max_new_tokens, do_sample=False)
    reply = _extract_reply(output)
    if not reply:
        raise RuntimeError(
            "Spark-X2.6-4B returned an empty answer; check the runtime setup "
            "and the generation budget"
        )
    return reply


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot (demo)"
    )
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
    parser.add_argument("--model", default=MODEL_ID, help="model id or local path")
    args = parser.parse_args(argv)

    reply = generate(args.question, generator=_load_pipeline(args.model))

    print("Question:", args.question)
    print()
    print(reply)
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
