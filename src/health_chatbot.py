"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

The app answers general health-education questions in plain language using a
locally served, OpenAI-compatible Spark-X2.6-4B endpoint (llama.cpp / vLLM /
SGLang / Spark-MLX-LLM server). It never diagnoses or prescribes, and it
redirects the user to a real clinician or emergency services when needed.

Run:
    python src/health_chatbot.py --question "My question"

Environment:
    SPARK_BASE_URL   base URL of the local server (default http://127.0.0.1:30000/v1)
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
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
DEFAULT_TIMEOUT = 120

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak tiga hari lalu, dan baru pulang "
    "dari perjalanan. Apa yang sebaiknya saya perhatikan?"
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "Consult a qualified clinician for your own situation."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant powered by Spark-X2.6-4B. "
    "Explain general health topics in short, plain sentences that a person "
    "without medical training can follow.\n"
    "Rules you must follow:\n"
    "1. Never give a diagnosis, a prescription, a dosage, or a treatment plan "
    "for an individual.\n"
    "2. Never claim to replace a clinician or a pharmacist; tell the user to "
    "see one for personal decisions.\n"
    "3. If the description suggests an emergency (for example chest pain, "
    "difficulty breathing, fainting, heavy bleeding, or thoughts of self-harm), "
    "tell the user to contact local emergency services immediately.\n"
    "4. Do not invent facts, statistics, drug names, or sources. If you are not "
    "sure, say so plainly.\n"
    "5. Give general background, questions the user could ask a clinician, and "
    "signs that mean they should seek care sooner.\n"
    "6. Answer in the language the user wrote in."
)


def build_messages(question: str, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_payload(messages, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS, temperature=0.3):
    """Build an OpenAI-compatible chat-completions payload."""
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }


def extract_reply(data):
    """Pull the assistant text out of a chat-completions response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected response shape from Spark-X2.6-4B") from exc

    content = (content or "").strip()
    if not content:
        raise RuntimeError("Spark-X2.6-4B returned an empty reply")
    return content


def chat(question: str, history=None, base_url=None, timeout=DEFAULT_TIMEOUT) -> str:
    """Send one question to the local Spark-X2.6-4B server and return the reply."""
    payload = build_payload(build_messages(question, history))
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            "Could not reach a Spark-X2.6-4B server at %s: %s. Start a local "
            "OpenAI-compatible server first (see the model card)." % (url, exc)
        ) from exc

    try:
        data = json.loads(body)
    except ValueError as exc:
        raise RuntimeError("Spark-X2.6-4B server returned invalid JSON") from exc

    return extract_reply(data)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health-information chatbot")
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="local OpenAI-compatible base URL")
    args = parser.parse_args(argv)

    print("Model: %s" % MODEL_ID)
    print("Question: %s" % args.question.strip())
    print("-" * 60)
    try:
        answer = chat(args.question, base_url=args.base_url)
    except (RuntimeError, ValueError) as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1

    print(answer)
    print("-" * 60)
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
