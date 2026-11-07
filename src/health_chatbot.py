"""Minimal Spark-X2.6-4B health-information chatbot.

HER Hack-Astron #7 (Spark-X2.6 Open Source AI for Good Challenge) entry:
an inclusive, plain-language patient-education assistant.

Spark-X2.6-4B runs behind a local OpenAI-compatible server (SGLang, vLLM,
llama.cpp or Spark-MLX-LLM). This module only builds the bounded prompt,
calls that endpoint and returns the text. It never diagnoses and never
prescribes: the system prompt forbids both and always redirects the reader
to a qualified clinician.

The model is text-only (see FAQ.md). No image path is offered here, so the
application cannot imply vision capability the model does not have.
"""

import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

MAX_QUESTION_CHARS = 4000
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people with low health literacy. "
    "Use short sentences and everyday words, and explain any medical term you "
    "must use. Give general, widely accepted health information only. "
    "Never diagnose, never prescribe, never give dosages, and never tell the "
    "reader to stop or change a treatment. "
    "If the question suggests an emergency (for example chest pain, difficulty "
    "breathing, heavy bleeding, fainting, or thoughts of self-harm), say clearly "
    "that the person should contact local emergency services right now. "
    "Always end by telling the reader to confirm anything important with a "
    "qualified clinician or pharmacist. "
    "If you are unsure, say that you are unsure instead of guessing."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and I still have a cough and fever. "
    "I do not have much money and I am not sure if this is serious."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    ``history`` is an optional list of ``{"role": ..., "content": ...}``
    turns. Only ``user`` and ``assistant`` turns are kept so a caller cannot
    replace the safety system prompt.
    """
    text = (question or "").strip()
    if not text:
        raise ValueError("question must not be empty")
    if len(text) > MAX_QUESTION_CHARS:
        text = text[:MAX_QUESTION_CHARS].rstrip() + " ..."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})
    return messages


def build_payload(messages):
    """Build the OpenAI-compatible request body (text-only, JSON mode off)."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.2,
        "stream": False,
    }


def parse_response(data):
    """Extract the assistant text, rejecting empty or malformed replies."""
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise ValueError("response has no choices")

    message = choice.get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise ValueError("model returned an empty answer")
    return content


def ask(question, history=None, base_url=None, api_key=None, timeout=REQUEST_TIMEOUT):
    """Send one question to a local Spark-X2.6-4B OpenAI-compatible endpoint."""
    messages = build_messages(question, history)
    payload = build_payload(messages)

    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = api_key or os.environ.get("SPARK_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError("Spark endpoint returned HTTP %s" % exc.code) from exc

    return parse_response(body)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a plain-language health question."
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args(argv)
    print(ask(args.question, base_url=args.base_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
