"""Minimal Spark-X2.6-4B health-information chatbot.

Dependency-free demo for the HER Hack-Astron #7 AI-for-Good entry:
a plain-language patient-education assistant. It never diagnoses and
never prescribes; it points the reader to a qualified clinician.

Run it against a local OpenAI-compatible Spark-X2.6-4B server:

    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py

With no argument the built-in DEMO_QUESTION is used. Any question can be
passed on the command line, for example:

    python src/health_chatbot.py "Apa tanda dehidrasi pada anak?"
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
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant at a community health desk. "
    "Explain health topics in simple, everyday language that a person with no "
    "medical training can follow, and keep answers short and concrete. "
    "Do not diagnose, do not name prescription medicines or doses, and never "
    "claim to replace a clinician. Always tell the reader which symptoms or "
    "timeframes mean they should see a doctor, and state clearly when a "
    "situation may be an emergency. Do not ask for or repeat personal "
    "identifiers such as names, addresses or ID numbers. If the question is "
    "unclear or too broad, ask one short clarifying question instead of "
    "guessing. Answer in the language the reader used."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a cough and a "
    "mild fever in the evenings. What should I know before I see a doctor?"
)


def build_messages(question, history=None):
    """Build the OpenAI-style chat messages for one turn."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": (question or "").strip()})
    return messages


def build_payload(question, history=None):
    """Build the request body sent to the Spark server."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
    }


def _post_json(url, payload, opener=None):
    """POST a JSON payload and return the decoded JSON response."""
    opener = opener or urllib.request.urlopen
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response = opener(request, timeout=REQUEST_TIMEOUT)
    try:
        body = response.read()
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            close()
    return json.loads(body.decode("utf-8"))


def extract_text(payload):
    """Pull the assistant text out of a chat-completions response."""
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("unexpected response shape from the Spark server")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty response from the Spark server")
    return content.strip()


def chat(question, base_url=None, opener=None):
    """Send one question to the local Spark-X2.6-4B endpoint."""
    base_url = (base_url or os.environ.get("SPARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    payload = build_payload(question)
    return extract_text(_post_json(base_url + "/chat/completions", payload, opener))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    try:
        answer = chat(question)
    except (urllib.error.URLError, ValueError, OSError) as exc:
        print("Spark-X2.6-4B request failed: %s" % exc, file=sys.stderr)
        print(
            "Start a local OpenAI-compatible server for %s and set SPARK_BASE_URL." % MODEL_ID,
            file=sys.stderr,
        )
        return 1
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
