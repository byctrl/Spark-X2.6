#!/usr/bin/env python3
"""Minimal public-interest health-information chatbot built on Spark-X2.6-4B.

The chatbot talks to a locally hosted Spark-X2.6-4B OpenAI-compatible endpoint
(SGLang / vLLM / llama.cpp / Spark-MLX-LLM). It gives plain-language education
only, never a diagnosis or a prescription, and always points the user to a
qualified clinician. Standard library only, no third-party dependencies.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
MAX_NEW_TOKENS = 1024
REQUEST_TIMEOUT = 120

DISCLAIMER = (
    "This is general health education, not a diagnosis or treatment plan. "
    "Please contact a qualified clinician or your local emergency service."
)

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information service. Answer in the user's language, in short plain "
    "sentences at roughly a 6th-grade reading level.\n"
    "Rules you must always follow:\n"
    "1. Explain the general meaning of symptoms, tests or terms; do not "
    "diagnose this person and do not name a definite disease as their "
    "condition.\n"
    "2. Do not prescribe medicines, doses, or home remedies presented as a "
    "treatment. Describe only general, widely accepted self-care and when to "
    "seek care.\n"
    "3. State clearly which warning signs mean 'seek urgent care now'.\n"
    "4. Never ask for or repeat identifying details (name, address, ID "
    "numbers, phone).\n"
    "5. If the question needs a clinician's judgement, say so and stop there.\n"
    "6. Never invent sources or citations."
)

# Demo scenario: a symptom question that must NOT receive a diagnosis.
DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and a fever that comes back every evening. What could it be and what "
    "should I do?"
)


def build_messages(question, history=None):
    """Build the chat messages for one turn, with the safety system prompt."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def post_chat(base_url, payload, timeout=REQUEST_TIMEOUT):
    """Default transport: POST /chat/completions to the local model server."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError("model endpoint returned HTTP %s: %s" % (exc.code, detail)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "cannot reach Spark-X2.6-4B at %s (%s); start the local server first"
            % (base_url, exc.reason)
        ) from exc


def parse_reply(data):
    """Extract the assistant text from an OpenAI-compatible response."""
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("unexpected response shape from model endpoint") from exc
    text = (text or "").strip()
    if not text:
        raise RuntimeError("model returned an empty answer")
    return text


def with_disclaimer(answer):
    """Ensure the fixed disclaimer is present exactly once."""
    answer = (answer or "").strip()
    if DISCLAIMER.lower() in answer.lower():
        return answer
    return answer + "\n\n" + DISCLAIMER


class HealthChatbot:
    """One-session chatbot wrapper around Spark-X2.6-4B."""

    def __init__(self, base_url=None, model=MODEL_ID, transport=post_chat, max_new_tokens=MAX_NEW_TOKENS):
        self.base_url = base_url or DEFAULT_BASE_URL
        self.model = model
        self.transport = transport
        self.max_new_tokens = max_new_tokens
        self.history = []

    def build_payload(self, messages):
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.2,
            "stream": False,
        }

    def ask(self, question):
        messages = build_messages(question, self.history)
        data = self.transport(self.base_url, self.build_payload(messages))
        answer = with_disclaimer(parse_reply(data))
        self.history.append({"role": "user", "content": question.strip()})
        self.history.append({"role": "assistant", "content": answer})
        return answer


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    base_url = DEFAULT_BASE_URL
    if args and args[0].startswith("--base-url="):
        base_url = args.pop(0).split("=", 1)[1]

    bot = HealthChatbot(base_url=base_url)
    if not args or args[0] in {"--demo", "-d"}:
        print("Demo question: %s\n" % DEMO_QUESTION)
        print(bot.ask(DEMO_QUESTION))
        return 0

    print(bot.ask(" ".join(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
