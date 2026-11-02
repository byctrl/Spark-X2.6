"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

Explains general health topics in plain language and points people to
professional care. It never diagnoses, prescribes, or replaces a clinician.

Usage:
    python src/health_chatbot.py                  # offline dry run
    python src/health_chatbot.py "my question"    # call a local model endpoint

The model endpoint is any OpenAI-compatible server (vLLM / SGLang / Ollama)
serving XHToken/Spark-X2.6-4B; override it with SPARK_BASE_URL.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_TIMEOUT = 60

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who need accessible "
    "health information. Use short sentences and plain language (about a "
    "6th-grade reading level). Explain what a symptom or document usually "
    "means, which questions to ask a clinician, and which kind of care "
    "setting fits the situation. You must not diagnose, prescribe, or give "
    "dosages. Always state that this is general education, not medical "
    "advice, and tell the user to contact a qualified clinician. If the user "
    "describes an emergency (chest pain, trouble breathing, heavy bleeding, "
    "fainting, thoughts of self-harm), tell them to contact local emergency "
    "services immediately. Never ask for names, IDs, or other personal data."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for three days since I got back "
    "from a trip. What should I do, and when should I see a doctor?"
)


def build_messages(question, history=None):
    """Build the chat-completions message list for one user question."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_payload(messages):
    """Wrap messages in the OpenAI-compatible request body."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
    }


def parse_completion(data):
    """Extract the assistant text from an OpenAI-compatible response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected completion payload: %r" % (data,))
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty completion content")
    return content.strip()


def generate_reply(question, history=None, base_url=None, timeout=DEFAULT_TIMEOUT):
    """Send the question to a locally hosted Spark-X2.6-4B endpoint."""
    base = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    body = json.dumps(build_payload(build_messages(question, history))).encode("utf-8")
    request = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "could not reach Spark-X2.6-4B at %s (%s); start a local "
            "OpenAI-compatible server or set SPARK_BASE_URL" % (base, exc)
        )
    return parse_completion(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health information demo")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the request payload instead of calling the model",
    )
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args(argv)

    payload = build_payload(build_messages(args.question))
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print("Question: " + args.question)
    print("")
    print(generate_reply(args.question, base_url=args.base_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
