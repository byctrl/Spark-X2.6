"""Minimal health-education chatbot powered by Spark-X2.6-4B.

Public-interest (AI for Good) demo for HER Hack-Astron #7: plain-language
health information for people who cannot easily reach a clinician.

The model runs behind an OpenAI-compatible endpoint (vLLM / SGLang / Ollama /
Spark-MLX-LLM, base URL configurable via ``SPARK_BASE_URL``). The bot gives
general education only: no diagnosis, no prescriptions, and it always points
the user to a qualified professional.

What the model does here: it receives a fixed safety system prompt plus the
user's question, and generates a short, plain-language educational answer that
names warning signs and the right next step. Everything else (input checks,
prompt assembly, response validation) is plain application code.

Run:
    python src/health_chatbot.py                 # built-in demo question
    python src/health_chatbot.py "my question"   # your own question
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a community health helpline.\n"
    "Rules you must follow:\n"
    "1. Use simple, everyday language. Explain any medical word you have to use.\n"
    "2. Do NOT diagnose, do NOT present a likely disease as a conclusion, and do "
    "NOT prescribe or change medicines, doses, or treatments.\n"
    "3. Do not ask for or repeat names, ID numbers, addresses, or other personal data.\n"
    "4. Give general information, then list warning signs that mean the person "
    "should seek urgent care, and advise seeing a qualified clinician about their "
    "own case.\n"
    "5. If the message suggests an emergency, say clearly: contact local emergency "
    "services now.\n"
    "6. If you are unsure, say so instead of guessing.\n"
    "Answer in the language the person used, in under 250 words."
)

DEMO_QUESTION = (
    "I came back from a trip three days ago and now I have a dry cough and a mild "
    "fever. What should I watch for, and when should I see a doctor?"
)


def build_messages(question: str) -> list:
    """Build the chat messages: safety system prompt first, question last."""
    text = (question or "").strip() if isinstance(question, str) else ""
    if not text:
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def build_payload(question: str, model: str = MODEL_ID, max_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Build the OpenAI-compatible request body for Spark-X2.6-4B."""
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    return {
        "model": model,
        "messages": build_messages(question),
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }


def extract_answer(response: dict) -> str:
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected model response shape") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def chat(
    question: str,
    base_url: str = None,
    timeout: int = REQUEST_TIMEOUT,
    max_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Send one question to the local Spark-X2.6-4B endpoint and return the answer."""
    root = (base_url or DEFAULT_BASE_URL).rstrip("/")
    url = root + "/chat/completions"
    body = json.dumps(build_payload(question, max_tokens=max_tokens)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError("could not reach Spark-X2.6 at %s: %s" % (url, exc)) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("model endpoint returned invalid JSON") from exc
    return extract_answer(payload)


def main(argv: list = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    print("Q: " + question)
    print("")
    print(chat(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
