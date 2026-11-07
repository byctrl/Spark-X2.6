"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

An educational, safety-bounded demo: it explains general health information in
plain language and points people to clinicians. It does not diagnose, prescribe,
or replace professional care, and it never dispatches alerts on its own.

Run against a local OpenAI-compatible Spark-X2.6 endpoint:

    export SPARK_BASE_URL=http://127.0.0.1:30000/v1
    python src/health_chatbot.py "I have had a cough and a mild fever for four days."
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant in a public-interest health information demo.\n"
    "Follow these rules for every reply:\n"
    "1. Use plain language: short sentences, common words, about a 6th-grade reading level.\n"
    "2. Give general health information only. Do not diagnose the reader, do not guess which "
    "condition they have, and do not prescribe medicines, doses, or treatments.\n"
    "3. List the warning signs that mean the reader should seek urgent in-person care.\n"
    "4. Tell the reader to confirm any decision with a qualified clinician or pharmacist.\n"
    "5. If the request is not general health education, say so briefly and name a better source.\n"
    "6. Never ask for names, ID numbers, addresses, or other identifying details.\n"
    "Reply in the language of the reader's question."
)

DISCLAIMER = (
    "\n\n---\n"
    "This is general education, not medical advice. "
    "Please confirm anything important with a qualified clinician or pharmacist."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days, since I came back from a trip. "
    "What general information should I know, and when should I see a clinician?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            raise ValueError("history entries need role user/assistant and non-empty content")
        messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS):
    if not messages:
        raise ValueError("messages must not be empty")
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False,
    }


def _post_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_answer(data):
    """Pull the answer text out of an OpenAI-compatible response."""
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response shape from the model server") from exc
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("the model returned an empty answer")
    return answer.strip()


def chat(messages, base_url=None, timeout=DEFAULT_TIMEOUT, transport=None):
    """Send messages to the model server and return the plain-text answer."""
    url = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    payload = build_payload(messages)
    send = transport or _post_json
    data = send(url + "/chat/completions", payload, timeout)
    return extract_answer(data)


def format_reply(answer):
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    return answer.strip() + DISCLAIMER


def ask(question, history=None, base_url=None, timeout=DEFAULT_TIMEOUT, transport=None):
    """Convenience wrapper: question in, disclaimer-bearing answer out."""
    messages = build_messages(question, history)
    return format_reply(chat(messages, base_url=base_url, timeout=timeout, transport=transport))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print("Model: " + MODEL_ID)
    print("Question: " + question)
    try:
        print("\nAnswer:")
        print(ask(question))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print("Could not reach the Spark-X2.6 server: %s" % exc, file=sys.stderr)
        print("Start a local OpenAI-compatible endpoint or set SPARK_BASE_URL.", file=sys.stderr)
        return 1
    except ValueError as exc:
        print("Request failed: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
