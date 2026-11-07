"""Minimal health-education chatbot built on Spark-X2.6-4B.

Answers general health questions in plain language for people who cannot
easily reach a clinician or read dense medical text. The assistant never
diagnoses, never prescribes and always points to professional care.

Spark-X2.6-4B is a text-only model. This module talks to any OpenAI-compatible
local server (vLLM, SGLang, Ollama, Spark-MLX-LLM) and uses only the Python
standard library, so it can be copied and run without installing anything.
"""

import argparse
import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
DEFAULT_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public health helpline. "
    "Explain health information in short, plain sentences that a person with "
    "no medical training can understand.\n"
    "Rules you must follow:\n"
    "1. Do not diagnose the user and do not present a likely disease as a conclusion.\n"
    "2. Do not prescribe medicines, doses or treatments.\n"
    "3. Say clearly when the user must contact a clinician, clinic or emergency service.\n"
    "4. List warning signs that need urgent care.\n"
    "5. If important information is missing, ask one short clarifying question instead of guessing.\n"
    "6. End by reminding the user that this is general information, not medical advice."
)

DEMO_QUESTION = (
    "I have had a dry cough and a mild fever for three days and I came back "
    "from a trip last week. What should I watch for and when should I see a doctor?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    ``history`` is an optional list of ``{"role": ..., "content": ...}``
    turns. Only ``user`` and ``assistant`` turns with text are kept.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages):
    """Build the OpenAI-compatible request body."""
    if not messages:
        raise ValueError("messages must not be empty")
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
        "stream": False,
    }


def build_request(messages, base_url=None, api_key=None):
    """Build a POST request to the local chat-completions endpoint."""
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    body = json.dumps(build_payload(messages)).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", "Bearer " + api_key)
    return request


def parse_response(data):
    """Extract the answer text, rejecting empty or malformed responses."""
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        raise ValueError("response contains no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response contains no answer text")
    return content.strip()


def ask(question, history=None, base_url=None, api_key=None, timeout=DEFAULT_TIMEOUT):
    """Send one question to Spark-X2.6-4B and return the answer text."""
    messages = build_messages(question, history)
    request = build_request(messages, base_url=base_url, api_key=api_key)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return parse_response(data)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a general health question (demo for the AI for Good challenge)."
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.environ.get("SPARK_API_KEY", ""))
    args = parser.parse_args(argv)
    print(ask(args.question, base_url=args.base_url, api_key=args.api_key))


if __name__ == "__main__":
    main()
