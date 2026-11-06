"""Minimal Spark-X2.6-4B health-information chatbot.

A small, dependency-free demo for the HER Hack-Astron #7 challenge: answer
general health-education questions in plain language. The assistant never
diagnoses or prescribes; it points people to a qualified clinician.

The model is served behind an OpenAI-compatible endpoint (llama.cpp, vLLM,
SGLang or Spark-MLX-LLM). Configure it with SPARK_BASE_URL / SPARK_API_KEY.

Usage:
    python src/health_chatbot.py --question "Saya batuk dan demam..."
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")

SYSTEM_PROMPT = (
    "You are a patient-education assistant powered by Spark-X2.6-4B. "
    "Explain general health topics in short, plain sentences that a person "
    "without medical training can understand. "
    "You must never diagnose, never name a prescription medicine or dose, and "
    "never replace a qualified clinician. "
    "State clearly that this is general information and that the person should "
    "contact a clinician or local emergency service for their own situation. "
    "If the question describes an emergency sign (for example trouble "
    "breathing, chest pain, heavy bleeding, or loss of consciousness), say "
    "first that they should seek emergency help now. "
    "Do not ask for names, ID numbers, addresses or other personal data."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago. Since yesterday I have "
    "a dry cough and a fever of about 38.2 C, and I feel very tired. "
    "What general information should I know, and when should I see a doctor?"
)


def build_messages(question: str) -> list[dict]:
    """Build the chat messages sent to Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_payload(
    question: str,
    model: str = MODEL_ID,
    max_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> dict:
    """Build the OpenAI-compatible chat-completions request body."""
    return {
        "model": model,
        "messages": build_messages(question),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def extract_answer(body: dict) -> str:
    """Pull the assistant text out of a chat-completions response."""
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("response is missing choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def request_chat(payload, base_url=None, api_key=None, timeout=120.0) -> str:
    """POST the payload to the local Spark-X2.6 endpoint and return the answer."""
    base_url = (base_url or BASE_URL).rstrip("/")
    api_key = api_key or API_KEY
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return extract_answer(body)


def answer(question: str, base_url=None, api_key=None, max_tokens=MAX_NEW_TOKENS) -> str:
    """Convenience wrapper: question in, plain-language answer out."""
    payload = build_payload(question, max_tokens=max_tokens)
    return request_chat(payload, base_url=base_url, api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a general health-education question."
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    print("Model: " + MODEL_ID)
    print("Question: " + args.question.strip())
    print("-" * 60)
    print(answer(args.question, base_url=args.base_url, max_tokens=args.max_tokens))
    print("-" * 60)
    print("General information only. Please consult a qualified clinician.")


if __name__ == "__main__":
    main()
