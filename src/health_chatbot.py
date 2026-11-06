"""Minimal health-information chatbot powered by Spark-X2.6-4B.

This is the small AI-for-Good demo requested in issue #16: a plain-language
patient-education assistant that runs against a locally hosted
Spark-X2.6-4B OpenAI-compatible endpoint (vLLM / SGLang / llama.cpp server).

The model only explains general health information in simple language. It is
explicitly not allowed to diagnose, prescribe, or replace a clinician, and the
application appends a fixed disclaimer to every answer so the safety boundary
lives in code rather than only in the prompt.

Usage:
    python src/health_chatbot.py
    python src/health_chatbot.py --question "..." --base-url http://127.0.0.1:30000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
MAX_HISTORY_TURNS = 8

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people with limited access to "
    "health information. Explain general health topics in short, simple "
    "sentences and avoid medical jargon; when a term is needed, define it."
    "\n\nSafety rules you must always follow:"
    "\n- Do not diagnose the user and do not tell them what disease or "
    "condition they have."
    "\n- Do not prescribe medicines, give dosages, or recommend starting or "
    "stopping a treatment."
    "\n- Do not claim the user can safely delay or skip professional care."
    "\n- Say clearly which symptoms need urgent in-person care."
    "\n- Encourage the user to confirm anything important with a qualified "
    "clinician, and remind them that you are not a doctor."
    "\n- If the question is outside general health education, say so briefly "
    "and suggest who could help."
)

DEMO_QUESTION = (
    "I have had a dry cough and a mild fever for four days, starting two "
    "weeks after I travelled abroad. I cannot see a doctor this week. What "
    "should I know and what warning signs should make me seek care sooner?"
)

DISCLAIMER = (
    "This is general health education, not medical advice or a diagnosis. "
    "Please talk to a qualified clinician about your own situation, and seek "
    "urgent care if your symptoms are severe or getting worse."
)

_ALLOWED_HISTORY_ROLES = ("user", "assistant")


def build_messages(question, history=None):
    """Build the chat message list sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        if not isinstance(history, (list, tuple)):
            raise ValueError("history must be a list of message dicts")
        for item in list(history)[-MAX_HISTORY_TURNS:]:
            if not isinstance(item, dict):
                raise ValueError("history entries must be dicts")
            role = item.get("role")
            content = item.get("content")
            if role not in _ALLOWED_HISTORY_ROLES:
                raise ValueError("history role must be 'user' or 'assistant'")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("history content must be a non-empty string")
            messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(messages, model=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body."""
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "stream": False,
    }


def parse_chat_completion(payload):
    """Extract the assistant text from an OpenAI-compatible response."""
    if not isinstance(payload, dict):
        raise ValueError("response payload must be a dict")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response payload has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def ensure_disclaimer(text):
    """Append the fixed safety disclaimer exactly once."""
    body = (text or "").strip()
    if DISCLAIMER in body:
        return body
    if not body:
        return DISCLAIMER
    return body + "\n\n" + DISCLAIMER


def ask(question, base_url=None, model=MODEL_ID, timeout=120.0, history=None):
    """Send one question to a local Spark-X2.6-4B endpoint and return the answer."""
    endpoint = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    payload = build_payload(build_messages(question, history), model=model)
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return ensure_disclaimer(parse_chat_completion(body))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spark-X2.6-4B health information chatbot")
    parser.add_argument("--question", default=DEMO_QUESTION, help="question to ask")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=MODEL_ID, help="served model name")
    parser.add_argument("--timeout", type=float, default=120.0, help="request timeout in seconds")
    args = parser.parse_args(argv)

    print("Q: " + args.question.strip())
    answer = ask(
        args.question,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
    )
    print("\nA: " + answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
