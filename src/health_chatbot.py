"""Minimal inclusive health-knowledge chatbot for HER Hack-Astron #7.

What the model does here: the application sends a short, bounded
patient-education prompt to **Spark-X2.6-4B** (XHToken/Spark-X2.6-4B),
served through any OpenAI-compatible local endpoint (vLLM / SGLang /
Ollama / Spark-MLX-LLM). Spark-X2.6-4B is a text-only model, so the app
keeps everything as text: it explains health topics in plain language
and points the reader to a clinician. It never diagnoses, never
prescribes, and never replaces professional care.

Run:
    python src/health_chatbot.py --question "What is a normal resting heart rate?"

The endpoint is configured with --base-url or the SPARK_BASE_URL
environment variable (default http://127.0.0.1:30000/v1).
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = """You are a patient-education assistant for a public-interest health service.

Your job:
- Explain health topics and common terms in short, plain sentences (about a 6th-grade reading level).
- State clearly, once, that you are not a clinician.
- Suggest when and where to seek professional care, including urgent-care warning signs.

Hard limits:
- Never diagnose, never present a disease as the user's confirmed condition, and never prescribe or give medicine doses.
- Never ask for or repeat identifying data such as names, ID numbers, addresses or phone numbers.
- If the question needs a clinician, say so plainly and stop; do not guess.
- If the question is not about health, say you can only help with health information.

Answer in the language the user writes in. Keep the answer under 200 words and end with one concrete next step."""

DEMO_QUESTION = (
    "I have had a dry cough and a 38 C fever since I came back from a trip "
    "five days ago. What should I pay attention to?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_payload(messages, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS):
    """Wrap messages in an OpenAI-compatible chat-completions payload."""
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def parse_reply(data):
    """Extract the assistant text from a chat-completions response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response shape from the Spark endpoint") from exc

    content = (content or "").strip()
    if not content:
        raise ValueError("Spark returned an empty answer")
    return content


def ask(question, history=None, base_url=DEFAULT_BASE_URL, model=MODEL_ID, timeout=REQUEST_TIMEOUT):
    """Send one question to Spark-X2.6-4B and return the plain-language answer."""
    payload = build_payload(build_messages(question, history), model=model)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError("could not reach the Spark endpoint at %s: %s" % (base_url, exc)) from exc
    return parse_reply(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Health knowledge chatbot on Spark-X2.6-4B")
    parser.add_argument("--question", default=DEMO_QUESTION, help="health question to ask")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible endpoint")
    parser.add_argument("--model", default=MODEL_ID, help="model id served by the endpoint")
    args = parser.parse_args(argv)
    print(ask(args.question, base_url=args.base_url, model=args.model))


if __name__ == "__main__":
    main()
