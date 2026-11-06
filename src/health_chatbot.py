"""Minimal Spark-X2.6-4B health-information chatbot.

Public-interest demo for HER Hack-Astron #7: an inclusive,
plain-language health education assistant. It does not diagnose,
prescribe, or replace a clinician; it points people to professional
care and to emergency services when needed.

The model is served by any OpenAI-compatible local endpoint
(SGLang / vLLM / llama.cpp server / Spark-MLX-LLM).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health "
    "information service. Explain health topics in plain, simple language "
    "at roughly a 6th-8th grade reading level, and be respectful of all "
    "genders, ages, disabilities and cultural backgrounds.\n"
    "Rules you must follow:\n"
    "1. You do NOT diagnose, prescribe, or give dosages, and you never tell "
    "the user which illness they have.\n"
    "2. Share only general, well-established health information and basic "
    "self-care.\n"
    "3. Always say when the user should see a qualified clinician, and list "
    "clear warning signs for the topic.\n"
    "4. For emergencies (chest pain, trouble breathing, heavy bleeding, "
    "thoughts of self-harm, signs of stroke, severe dehydration), tell the "
    "user to contact local emergency services immediately.\n"
    "5. If information is uncertain or depends on the region, say so instead "
    "of guessing.\n"
    "6. Do not ask for or repeat personal identifiers, and remind the user "
    "not to share private medical records.\n"
    "End every answer with a short reminder that this is general education, "
    "not medical advice."
)

DEMO_QUESTION = (
    "I have had a cough and a mild fever for four days, after coming back "
    "from a trip. What should I know before I see a doctor, and when should "
    "I get help right away?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
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


def build_payload(messages, max_new_tokens=MAX_NEW_TOKENS):
    """Build the OpenAI-compatible request body."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def extract_text(body):
    """Return the assistant text from a chat-completion response body."""
    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices:
        raise RuntimeError("Spark endpoint returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Spark endpoint returned an empty answer")
    return content.strip()


def complete(messages, base_url=None, api_key=None, timeout=REQUEST_TIMEOUT, opener=None):
    """Send messages to the local Spark endpoint and return the answer text."""
    url = (base_url or BASE_URL).rstrip("/") + "/chat/completions"
    data = json.dumps(build_payload(messages)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + (api_key or API_KEY),
        },
    )
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError("Spark endpoint returned HTTP %s" % exc.code) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot reach the Spark endpoint at %s; start a local "
            "OpenAI-compatible server first." % url
        ) from exc
    return extract_text(body)


def ask(question, history=None, **kwargs):
    """One-turn convenience wrapper: question in, plain-language answer out."""
    return complete(build_messages(question, history), **kwargs)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print("Q:", question)
    print()
    print(ask(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
