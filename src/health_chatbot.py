"""Minimal health-education chatbot powered by Spark-X2.6-4B.

Safety-bounded: it gives general patient education only and never diagnoses,
prescribes or replaces a clinician. It talks to a locally hosted
OpenAI-compatible Spark-X2.6-4B endpoint (vLLM / SGLang / Ollama / MLX server).

Usage:
    python src/health_chatbot.py "batuk kering dan demam sejak 4 hari"
    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in plain, "
    "respectful language a non-specialist can understand. Prefer short paragraphs "
    "and bullet points.\n"
    "Hard rules:\n"
    "- Do not diagnose, prescribe, or give dosages. Say plainly that you cannot do this.\n"
    "- Do not replace a clinician. Encourage the reader to contact a qualified health "
    "worker or a local emergency service, especially for severe or worsening symptoms.\n"
    "- Do not ask for or repeat identifying personal data.\n"
    "- Say when information is uncertain or may vary by region and health system.\n"
    "- Keep answers short: what the symptoms may relate to in general, what to watch for, "
    "and which type of professional to consult."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam 38.2 C sejak 4 hari lalu, baru pulang dari perjalanan. "
    "Apa yang sebaiknya saya lakukan?"
)


def build_messages(question: str, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages, max_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.2):
    """Build an OpenAI-compatible chat completion payload."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def parse_response(data) -> str:
    """Extract the assistant text from an OpenAI-compatible response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected response from the Spark-X2.6 endpoint") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Spark-X2.6 returned an empty answer")
    return content.strip()


def _post_json(url: str, payload, timeout: int):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask(question: str, base_url=None, history=None, poster=None, timeout: int = REQUEST_TIMEOUT) -> str:
    """Ask Spark-X2.6-4B one question and return the plain-language answer."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    base = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    payload = build_payload(build_messages(question, history))
    poster = poster or _post_json
    return parse_response(poster(base + "/chat/completions", payload, timeout))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health-education chatbot demo")
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=None, help=f"OpenAI-compatible endpoint (default {DEFAULT_BASE_URL})")
    args = parser.parse_args(argv)
    try:
        print(ask(args.question, base_url=args.base_url))
    except Exception as exc:  # demo CLI: report and exit non-zero
        print(f"error: {exc}")
        print(
            "hint: start a local OpenAI-compatible Spark-X2.6-4B server and point "
            f"SPARK_BASE_URL at it (default {DEFAULT_BASE_URL})"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
