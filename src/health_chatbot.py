"""Minimal health-education chatbot built on Spark-X2.6-4B.

Public-interest demo for the HER Hack-Astron #7 challenge (health and
accessibility). The assistant explains health topics in plain language and
never diagnoses or prescribes; it always points the reader to a qualified
clinician.

Backends:
  * ``openai``       - any OpenAI-compatible server (vLLM / SGLang / llama.cpp)
                       serving ``XHToken/Spark-X2.6-4B``.
  * ``transformers`` - local ``transformers`` pipeline (optional dependency).

Usage:
    python src/health_chatbot.py
    python src/health_chatbot.py --backend openai "What is a fever?"
    python src/health_chatbot.py --backend transformers "What is a fever?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
MAX_QUESTION_CHARS = 4000
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who have limited access "
    "to health information. Explain in plain, respectful language at roughly a "
    "6th-grade reading level, using short sentences and everyday words. "
    "You must NOT diagnose, must NOT prescribe, and must NOT give medicine "
    "doses; you must not claim to replace a clinician. "
    "If the message suggests an emergency (chest pain, difficulty breathing, "
    "severe bleeding, fainting, sudden weakness, or thoughts of self-harm), "
    "tell the user to contact local emergency services right now. "
    "Otherwise give general educational information, list two or three "
    "questions the user can ask a health worker, and end with a reminder to "
    "see a qualified health worker. Never invent sources, statistics or "
    "guidelines. Reply in the language the user wrote in."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and now I have a dry cough and a "
    "fever in the evening. What could it be and what should I do?"
)

DISCLAIMER = (
    "Educational information only - not medical advice. "
    "Please see a qualified health worker about your own situation."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    ``history`` is an optional list of ``{"role": ..., "content": ...}``
    turns with role ``user`` or ``assistant``. System turns are rejected so
    the safety prompt cannot be overridden by caller input.
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
            raise ValueError("history turns need role user/assistant and text content")
        messages.append({"role": role, "content": content})

    messages.append(
        {"role": "user", "content": question.strip()[:MAX_QUESTION_CHARS]}
    )
    return messages


def build_prompt(question, history=None, tokenizer=None):
    """Render a plain prompt, using the tokenizer chat template if available."""
    messages = build_messages(question, history)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    body = "\n\n".join(
        "{0}: {1}".format(message["role"].upper(), message["content"])
        for message in messages
    )
    return body + "\n\nASSISTANT:"


def format_answer(text):
    """Validate the model output and append the fixed safety disclaimer."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("model returned an empty answer")
    return text.strip() + "\n\n" + DISCLAIMER


def chat_openai(
    question,
    history=None,
    base_url=DEFAULT_BASE_URL,
    model=MODEL_ID,
    timeout=120.0,
    max_tokens=MAX_NEW_TOKENS,
    opener=urllib.request.urlopen,
):
    """Ask a local OpenAI-compatible Spark-X2.6-4B endpoint."""
    payload = {
        "model": model,
        "messages": build_messages(question, history),
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response from the model server") from exc
    return format_answer(content)


def load_pipeline(model_id=MODEL_ID):
    """Load a local transformers pipeline (heavy optional dependency)."""
    from transformers import pipeline

    return pipeline("text-generation", model=model_id, device_map="auto")


def chat_transformers(
    question,
    history=None,
    generator=None,
    model_id=MODEL_ID,
    max_new_tokens=MAX_NEW_TOKENS,
):
    """Ask Spark-X2.6-4B through a local transformers pipeline."""
    if generator is None:
        generator = load_pipeline(model_id)
    prompt = build_prompt(question, history)
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    if isinstance(output, list) and output:
        text = output[0].get("generated_text", "")
    else:
        text = output
    return format_answer(text)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plain-language health education chatbot on Spark-X2.6-4B"
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="question to ask; defaults to the built-in demo question",
    )
    parser.add_argument(
        "--backend", choices=("openai", "transformers"), default="openai"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip() or DEMO_QUESTION

    try:
        if args.backend == "openai":
            answer = chat_openai(
                question,
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_new_tokens,
            )
        else:
            answer = chat_transformers(
                question,
                model_id=args.model,
                max_new_tokens=args.max_new_tokens,
            )
    except (urllib.error.URLError, ValueError, ImportError) as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
