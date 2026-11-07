"""Minimal Spark-X2.6-4B health-information chatbot.

Accessibility / health entry for HER Hack-Astron #7 (AI for Good).

The assistant gives plain-language *patient education* only: it never
diagnoses, never prescribes, and always points the reader to a qualified
clinician. It runs entirely against a locally hosted Spark-X2.6-4B that
exposes an OpenAI-compatible endpoint, e.g.

    python -m sglang.launch_server \
        --model-path XHToken/Spark-X2.6-4B --port 30000

    SPARK_BASE_URL=http://127.0.0.1:30000/v1 \
        python -m src.health_chatbot

No third-party dependency is required: the HTTP call uses ``urllib``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_TIMEOUT = 120.0

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain health information in plain, "
    "respectful language that a person with no medical training can understand. "
    "Never diagnose, never name a specific medicine or dose, and never replace a "
    "clinician. Always say clearly when professional care is needed, and list the "
    "warning signs that require urgent attention. If the question is not general "
    "health education, say so and suggest the right kind of professional."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago. Since yesterday I have a dry "
    "cough and a mild fever (38.1 C). I have no chest pain. What should I know, and "
    "when should I see a doctor?"
)

DISCLAIMER = (
    "This is general health education, not medical advice. "
    "Please contact a qualified clinician about your own situation."
)


class SparkClientError(RuntimeError):
    """Raised when the local Spark-X2.6-4B endpoint cannot be used."""


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    ``history`` is an optional list of ``{"role": ..., "content": ...}``
    turns; only well-formed ``user`` / ``assistant`` turns are kept.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question, history=None, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS):
    """Build the JSON payload for an OpenAI-compatible /chat/completions call."""
    return {
        "model": model,
        "messages": build_messages(question, history),
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }


def build_prompt(messages):
    """Flatten messages into one prompt, for completion-style endpoints."""
    parts = []
    for message in messages:
        parts.append("<|{0}|>\n{1}".format(message["role"], message["content"]))
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def clean_reply(text, max_chars=8000):
    """Normalise a raw model reply for display."""
    if not isinstance(text, str):
        raise ValueError("reply must be a string")
    reply = text.strip()
    for prefix in ("<|assistant|>", "assistant:", "Assistant:"):
        if reply.startswith(prefix):
            reply = reply[len(prefix):].strip()
    if len(reply) > max_chars:
        reply = reply[:max_chars].rstrip() + " ..."
    return reply


def with_disclaimer(reply):
    """Append the fixed safety disclaimer when it is not already present."""
    body = clean_reply(reply)
    if not body:
        return DISCLAIMER
    if DISCLAIMER in body:
        return body
    return body + "\n\n" + DISCLAIMER


class SparkClient(object):
    """Tiny OpenAI-compatible client for a locally served Spark-X2.6-4B."""

    def __init__(self, base_url=None, model=MODEL_ID, timeout=DEFAULT_TIMEOUT):
        self.base_url = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.timeout = timeout

    def endpoint(self):
        return self.base_url + "/chat/completions"

    def chat(self, question, history=None):
        payload = build_payload(question, history, model=self.model)
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint(),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            raise SparkClientError(
                "Could not reach Spark-X2.6-4B at {0}: {1}".format(self.endpoint(), exc)
            )
        except ValueError as exc:
            raise SparkClientError("Endpoint returned invalid JSON: {0}".format(exc))

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise SparkClientError("Unexpected response shape: {0}".format(body))
        return with_disclaimer(content)


def answer(question, base_url=None, history=None):
    """Convenience wrapper used by the CLI and by callers embedding the demo."""
    return SparkClient(base_url=base_url).chat(question, history=history)


def main():
    client = SparkClient()
    print("Model: {0}".format(client.model))
    print("Endpoint: {0}".format(client.endpoint()))
    print("Question: {0}".format(DEMO_QUESTION))
    print("-" * 60)
    try:
        print(client.chat(DEMO_QUESTION))
    except SparkClientError as exc:
        print("Error: {0}".format(exc))
        print("Start a local OpenAI-compatible Spark-X2.6-4B server first.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
