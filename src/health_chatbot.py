"""Minimal Spark-X2.6-4B chatbot for inclusive health education.

Demo application for the HER Hack-Astron #7 "AI for Good" challenge:
a plain-language health-information assistant built on XHToken/Spark-X2.6-4B.

What the model does here: it turns a layperson's health question into a short,
respectful explanation plus the warning signs that require professional care.
It never diagnoses, never prescribes and never asks for personal data.

Run against a local OpenAI-compatible server (vLLM, SGLang, llama.cpp, LM Studio):

    python src/health_chatbot.py "cough and fever three days after travelling"

Environment variables:
    SPARK_BASE_URL  OpenAI-compatible base URL (default http://127.0.0.1:30000/v1)
    SPARK_API_KEY   optional bearer token for the local server
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_TIMEOUT = 120
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")

SYSTEM_PROMPT = (
    "You are a patient-education assistant inside a public-interest health service. "
    "Explain health information in simple, respectful, inclusive language for a "
    "reader with no medical training. Any gender, any literacy level, any device. "
    "Rules you must always follow:\n"
    "1. Never diagnose, never name a definitive condition, never prescribe or dose medicine.\n"
    "2. Say clearly that this is general education and not medical advice.\n"
    "3. List the warning signs that mean the person should seek urgent professional care.\n"
    "4. Never ask for, repeat or infer personal identifiers; do not keep personal data.\n"
    "5. If the question is outside general health education, say so and point to a clinician.\n"
    "6. Answer in the language of the question, in short sentences, without jargon.\n"
    "7. End with one short line naming who to contact: a local clinician or the local "
    "emergency number."
)

DEMO_QUESTION = (
    "Saya batuk kering dan demam ringan sejak tiga hari lalu, beberapa hari "
    "setelah pulang dari perjalanan. Apa yang sebaiknya saya perhatikan?"
)

DISCLAIMER = (
    "General health education only - not a diagnosis and not a prescription. "
    "Contact a qualified clinician about your own situation."
)


def build_messages(question, history=None):
    """Build the chat messages: system guardrails, prior turns, then the question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            raise ValueError("history turns must be dictionaries")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError("history turns need role in {user, assistant} and string content")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(question, history=None, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body sent to the local Spark server."""
    return {
        "model": model,
        "messages": build_messages(question, history=history),
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
    }


def http_transport(payload, base_url=DEFAULT_BASE_URL, api_key=None, timeout=DEFAULT_TIMEOUT):
    """Default transport: POST the payload to a local OpenAI-compatible endpoint."""
    url = str(base_url).rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    key = api_key or os.environ.get("SPARK_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError("Spark endpoint returned HTTP %s: %s" % (exc.code, detail)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot reach the Spark endpoint at %s. Start a local OpenAI-compatible "
            "server serving %s, or set SPARK_BASE_URL. (%s)" % (url, MODEL_ID, exc.reason)
        ) from exc


def extract_text(response):
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        choice = response["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("response has no choices[0]") from exc
    message = (choice or {}).get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def health_chat(question, history=None, model=MODEL_ID, base_url=DEFAULT_BASE_URL, transport=None):
    """Ask Spark-X2.6-4B one health-education question and return the plain answer."""
    send = transport or http_transport
    payload = build_payload(question, history=history, model=model)
    return extract_text(send(payload, base_url))


def render_answer(answer, disclaimer=DISCLAIMER):
    """Attach the mandatory non-medical-advice notice for display."""
    return "%s\n\n---\n%s" % (answer.strip(), disclaimer)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    print("Model: %s" % MODEL_ID)
    print("Endpoint: %s" % DEFAULT_BASE_URL)
    print("Question: %s\n" % question)
    print(render_answer(health_chat(question)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
