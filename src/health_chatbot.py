"""Minimal Spark-X2.6-4B health-information chatbot demo.

Public-interest (health) entry for HER Hack-Astron #7.
Text-only: Spark-X2.6 does not accept images as model input; see FAQ.md.
The model is reached through an OpenAI-compatible endpoint, so the demo
works with vLLM / SGLang / llama.cpp / Spark-MLX-LLM local serving.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in plain language "
    "that a non-specialist can read. You must not diagnose, must not prescribe, and must not "
    "recommend a treatment, medicine or dose. Point out clearly when a symptom needs urgent "
    "in-person care, and always tell the reader to confirm anything important with a qualified "
    "clinician. If the information given is not enough, say what is missing instead of guessing."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have had a dry cough and "
    "a fever in the evening. What should I know before I see a doctor?"
)


def build_messages(question: str) -> list[dict[str, str]]:
    """Build a bounded system+user message list for the model."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def build_payload(question: str, model: str = MODEL_ID) -> dict:
    """Build the OpenAI-compatible request body sent to Spark-X2.6-4B."""
    return {
        "model": model,
        "messages": build_messages(question),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
    }


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def chat(question: str, base_url: str | None = None, timeout: float = 120.0) -> str:
    """Send one question to a locally served Spark-X2.6-4B and return its answer."""
    root = (base_url or BASE_URL).rstrip("/")
    payload = build_payload(question)
    try:
        response = _post_json(root + "/chat/completions", payload, timeout)
    except OSError as exc:
        raise RuntimeError(
            "Could not reach a Spark-X2.6-4B OpenAI-compatible endpoint at "
            f"{root}. Start a local server (see README quickstart) or set "
            f"SPARK_BASE_URL. Underlying error: {exc}"
        ) from exc
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response shape from {root}: {response!r}") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The model returned an empty answer.")
    return content.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a plain-language health question."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question text; the built-in demo question is used if omitted",
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip() or DEMO_QUESTION
    print(f"Model: {MODEL_ID}")
    print(f"Question: {question}\n")
    try:
        print(chat(question, base_url=args.base_url))
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
