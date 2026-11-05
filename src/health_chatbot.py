"""Minimal Spark-X2.6-4B health-information chatbot.

Public-interest / accessibility demo: answers general health-education
questions in plain language through a locally hosted OpenAI-compatible
Spark-X2.6 endpoint.

This is not a medical device. It does not diagnose, prescribe or triage.
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
DEFAULT_TIMEOUT = 120.0

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain general health topics in "
    "plain language at roughly a sixth-grade reading level.\n"
    "Rules you must always follow:\n"
    "1. Do not diagnose, do not tell the user that they have a specific "
    "disease, and do not recommend prescription medicines or dosages.\n"
    "2. Do not tell the user to delay or avoid professional care.\n"
    "3. Encourage the user to contact a qualified clinician or a local "
    "emergency service about their own symptoms, and say clearly when a "
    "symptom pattern may need urgent care.\n"
    "4. Say when you are unsure, and never invent sources or citations.\n"
    "5. Keep answers short: a few short paragraphs or a short list."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a cough and a "
    "low fever in the evenings. What should I know before I see a doctor?"
)

DEMO_ANSWER_NOTE = (
    "Demo scenario only. The reply is general education, not a diagnosis."
)


def build_messages(question, history=None):
    """Build the OpenAI-style message list for one turn."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            raise ValueError(
                "history entries need role=user|assistant and non-empty content"
            )
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_payload(messages, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS):
    """Build the chat-completions request body."""
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False,
    }


def extract_answer(response):
    """Pull the assistant text out of a chat-completions response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("endpoint returned no assistant message") from exc
    content = (content or "").strip()
    if not content:
        raise ValueError("endpoint returned an empty assistant message")
    return content


class HealthChatbot:
    """Thin client for a local Spark-X2.6-4B OpenAI-compatible server."""

    def __init__(
        self,
        base_url=DEFAULT_BASE_URL,
        model=MODEL_ID,
        timeout=DEFAULT_TIMEOUT,
    ):
        self.base_url = (base_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        self.model = model
        self.timeout = timeout

    def _post(self, payload):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "could not reach the Spark endpoint at %s: %s"
                % (self.base_url, exc)
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("endpoint returned invalid JSON") from exc

    def ask(self, question, history=None):
        """Return the model's plain-language answer for ``question``."""
        messages = build_messages(question, history)
        response = self._post(build_payload(messages, model=self.model))
        return extract_answer(response)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a general health-education question."
    )
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)

    bot = HealthChatbot(
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
    )

    print("Q: " + args.question)
    print()
    try:
        print(bot.ask(args.question))
    except (ValueError, RuntimeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print()
    print(DEMO_ANSWER_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
