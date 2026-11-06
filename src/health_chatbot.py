"""Minimal health-information chatbot demo for the HER Hack-Astron #7 challenge.

Uses Spark-X2.6-4B (MODEL_ID) through a local OpenAI-compatible endpoint
(SGLang / vLLM / Ollama, see the model card). Standard library only.

The assistant is deliberately bounded: it explains health information in
plain language and never diagnoses or prescribes.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in plain "
    "language at about a 6th-grade reading level for people with limited medical "
    "literacy. Never diagnose, never prescribe, never suggest a dose, and never tell "
    "the user to start, stop or change a treatment. Do not claim to replace a "
    "clinician. State the limits of the information you give, list the warning signs "
    "that need urgent care, and tell the user to confirm everything with a qualified "
    "clinician or pharmacist. If the question suggests an emergency, tell the user to "
    "contact local emergency services first. Reply in the language of the question."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a cough and a "
    "fever that comes back every evening. I have no insurance. What should I know "
    "before I talk to a doctor?"
)


def build_messages(question: str, system_prompt: str = SYSTEM_PROMPT) -> list:
    """Build the bounded chat messages for one question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def build_payload(
    question: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    model_id: str = MODEL_ID,
) -> dict:
    """Build the OpenAI-compatible request body."""
    return {
        "model": model_id,
        "messages": build_messages(question),
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
    }


def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_answer(response) -> str:
    """Pull the assistant text out of an OpenAI-compatible response."""
    if isinstance(response, str):
        answer = response.strip()
    else:
        try:
            answer = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("unexpected response shape from the Spark endpoint") from exc
        answer = (answer or "").strip()
    if not answer:
        raise ValueError("model returned an empty answer")
    return answer


def chat(
    question: str,
    base_url: str = None,
    transport=None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Ask Spark-X2.6-4B one question and return its plain-language answer."""
    base = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    url = base + "/chat/completions"
    payload = build_payload(question, max_new_tokens=max_new_tokens)
    send = transport or _post_json
    return extract_answer(send(url, payload))


def main() -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health-information chatbot")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()
    print("model:", MODEL_ID)
    print("question:", args.question)
    print("-" * 60)
    print(chat(args.question, base_url=args.base_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
