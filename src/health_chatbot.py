"""Minimal Spark-X2.6-4B health-education chatbot.

Public-interest demo for HER Hack-Astron #7: answers general health
questions in plain language for people with limited access to reliable
health information. The model is used as a *language and explanation*
layer only. It never diagnoses, prescribes, or replaces a clinician, and
the application always appends a fixed safety notice in code (not in the
prompt) so the disclaimer cannot be talked away by user input.

Run against a local OpenAI-compatible server (SGLang / vLLM / llama.cpp):

    python src/health_chatbot.py --demo
    python src/health_chatbot.py --question "..." \
        --base-url http://127.0.0.1:30000/v1

The ``openai`` package is imported lazily, only when a real request is
made, so the module and its tests run with the standard library alone.
"""

from __future__ import annotations

import argparse
import os

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3

DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
DEFAULT_API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")

SAFETY_NOTICE = (
    "This is general health education, not a diagnosis or a prescription. "
    "If symptoms are severe, sudden, or getting worse, contact a qualified "
    "clinician or your local emergency number now."
)

SYSTEM_PROMPT = """You are a patient-education assistant for people who may have \
limited access to health care, limited literacy, or limited local-language \
health material. You are not a doctor and you cannot examine anyone.

Rules you must follow:
1. Use short sentences and everyday words. Avoid medical jargon; if a term \
   is unavoidable, explain it in one clause.
2. Explain general background: what the symptom or topic usually means, \
   which questions a clinician would ask, what information the person \
   should prepare, and which red-flag signs mean "seek care now".
3. Never give a diagnosis, never name a specific medicine, dose, or \
   treatment plan, and never say a condition is "nothing to worry about".
4. Never claim certainty. Say plainly what is unknown without an \
   examination or test.
5. If the description suggests an emergency (for example chest pain, \
   trouble breathing, heavy bleeding, fainting, signs of stroke, severe \
   pain, or a very sick infant), tell the person to seek emergency care \
   immediately and stop the general explanation there.
6. Do not ask for names, addresses, ID numbers, or other identifying data. \
   Suggest the person avoid sharing personal details.
7. End with 1-3 concrete next steps the person can take locally.

If the question is outside health, say so briefly and steer back to health \
information."""

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago and now I have a dry "
    "cough and a mild fever. I do not have a regular doctor. What should I "
    "know before I decide where to get help?"
)


def build_messages(question: str, history=None) -> list:
    """Build the chat payload sent to Spark-X2.6-4B.

    ``history`` is an optional list of ``{"role": ..., "content": ...}``
    turns from earlier in the conversation; it is not mutated.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant"):
            raise ValueError("history role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("history content must be a non-empty string")
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def default_client(base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY):
    """Create an OpenAI-compatible client for a locally served Spark-X2.6-4B."""
    from openai import OpenAI  # imported here so tests need no dependency

    return OpenAI(base_url=base_url, api_key=api_key)


def generate_reply(
    question: str,
    history=None,
    client=None,
    model: str = MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Ask Spark-X2.6-4B for a plain-language health explanation."""
    messages = build_messages(question, history)
    if client is None:
        client = default_client()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=TEMPERATURE,
    )

    content = response.choices[0].message.content
    text = (content or "").strip()
    if not text:
        raise RuntimeError(
            "Spark returned an empty answer; check the endpoint, the model id "
            "and the token budget before showing this to a user."
        )
    return text


def answer(question: str, history=None, client=None, model: str = MODEL_ID):
    """Return the model answer plus the code-enforced safety notice."""
    reply = generate_reply(question, history=history, client=client, model=model)
    return {"model": model, "answer": reply, "safety_notice": SAFETY_NOTICE}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health education chatbot")
    parser.add_argument("--question", default=None, help="health question to ask")
    parser.add_argument("--demo", action="store_true", help="use the built-in demo question")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    question = DEMO_QUESTION if (args.demo or not args.question) else args.question
    client = default_client(base_url=args.base_url)
    reply = generate_reply(
        question,
        client=client,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
    )

    print("Q:", question)
    print()
    print("A:", reply)
    print()
    print(SAFETY_NOTICE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
