"""Minimal Spark-X2.6-4B health-information chatbot.

Patient-education helper for the HER Hack-Astron #7 AI-for-Good challenge.
The model explains health topics in plain language; it never diagnoses,
prescribes, or replaces a clinician.

Run the demo (requires transformers + model weights):
    python src/health_chatbot.py --demo

Ask a question:
    python src/health_chatbot.py --question "What is a normal resting heart rate?"
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_LANGUAGE = "en"

SYSTEM_PROMPT = (
    "You are a patient-education assistant powered by Spark-X2.6-4B. "
    "Explain health information in plain language for a general audience. "
    "You must not diagnose, prescribe, or recommend a specific medicine or dose. "
    "Always say when a concern needs a qualified clinician, and tell the reader to "
    "contact emergency services for severe or rapidly worsening symptoms. "
    "If the question is unclear, ask one short clarifying question. "
    "Keep answers factual, calm, and free of speculation."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago and now I have a dry cough "
    "and a mild fever. What should I pay attention to, and when should I see a doctor?"
)


def build_messages(question: str, language: str = DEFAULT_LANGUAGE) -> List[Dict[str, str]]:
    """Return the chat messages sent to Spark-X2.6-4B."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must not be empty")

    system = SYSTEM_PROMPT
    if language and language.lower().startswith("zh"):
        system += " Reply in Simplified Chinese."
    else:
        system += " Reply in the same language as the question when possible."

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def render_prompt(messages: List[Dict[str, str]], tokenizer: Optional[Any] = None) -> str:
    """Render chat messages with the tokenizer template, or a plain fallback."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    lines = ["{0}: {1}".format(m["role"], m["content"]) for m in messages]
    lines.append("assistant:")
    return "\n".join(lines)


def generate(
    question: str,
    model_id: str = MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Generate a plain-language answer from Spark-X2.6-4B."""
    messages = build_messages(question, language=language)

    # Imported lazily so the module and its tests run without model weights.
    from transformers import pipeline

    generator = pipeline("text-generation", model=model_id, device_map="auto")
    prompt = render_prompt(messages, tokenizer=getattr(generator, "tokenizer", None))
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return output[0]["generated_text"].strip()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health information chatbot")
    parser.add_argument("--question", help="health question to ask")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo question")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    args = parser.parse_args(argv)

    if args.demo:
        question = DEMO_QUESTION
    elif args.question:
        question = args.question
    else:
        parser.error("provide --question or --demo")
        return 2

    answer = generate(
        question,
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        language=args.language,
    )
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
