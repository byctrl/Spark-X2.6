"""Minimal Spark-X2.6-4B health-information chatbot (AI for Good demo).

The chatbot is a patient-education helper: it explains general health
information in plain language and always defers diagnosis, treatment and
dosing to a qualified clinician. It talks to a locally hosted
OpenAI-compatible Spark-X2.6-4B endpoint, for example:

    python -m sglang.launch_server --model-path XHToken/Spark-X2.6-4B --port 30000
    python src/health_chatbot.py "Batuk kering dan demam sejak tiga hari lalu"

Environment variables:
    SPARK_BASE_URL   OpenAI-compatible base URL (default http://127.0.0.1:30000/v1)

No third-party dependencies: only the Python standard library is used.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health chatbot. "
    "Answer in the same language as the user's question, in short plain-language "
    "paragraphs and bullet points that a person with low health literacy can follow.\n"
    "Scope and rules:\n"
    "- Explain general health information only: what a symptom may be associated "
    "with, when and where to seek care, and what to prepare for a clinic visit.\n"
    "- Never diagnose this person, never name a definitive condition for them, and "
    "never prescribe, recommend a dose, or advise stopping a prescribed medicine.\n"
    "- Never ask for or repeat names, ID numbers, addresses, phone numbers or other "
    "identifying details.\n"
    "- If the description may be an emergency (for example chest pain, trouble "
    "breathing, heavy bleeding, confusion, fainting, or a newborn with fever), tell "
    "the user to contact local emergency services right now and keep the rest short.\n"
    "- End with one sentence reminding the user that this is general education and a "
    "qualified clinician should confirm any decision.\n"
    "- If the question is not about health, say briefly that you only cover health "
    "education and invite a health question."
)

DEMO_QUESTION = (
    "Saya demam dan batuk kering sejak tiga hari lalu, setelah pulang dari "
    "perjalanan. Kapan sebaiknya saya ke klinik?"
)

DISCLAIMER = (
    "\n\n---\n"
    "Edukasi umum, bukan diagnosis atau resep. "
    "General education only - not a diagnosis or a prescription. "
    "Please confirm with a qualified clinician."
)


def build_messages(question):
    """Return the chat messages sent to Spark for one user question."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_payload(messages, model=MODEL_ID, max_tokens=MAX_NEW_TOKENS):
    """Build a bounded, non-streaming OpenAI-compatible chat payload."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }


def extract_answer(response):
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unexpected response shape from the Spark endpoint") from exc
    content = (content or "").strip()
    if not content:
        raise ValueError("Spark returned an empty answer")
    return content


def with_disclaimer(answer):
    """Append the fixed safety disclaimer once."""
    answer = (answer or "").strip()
    if not answer:
        raise ValueError("answer must be a non-empty string")
    if DISCLAIMER.strip() in answer:
        return answer
    return answer + DISCLAIMER


def _post_json(url, payload, timeout=REQUEST_TIMEOUT):
    """POST JSON with the standard library and return the decoded response."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def chat(question, base_url=None, transport=None, timeout=REQUEST_TIMEOUT):
    """Send one question to Spark-X2.6-4B and return the plain-language answer."""
    base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    payload = build_payload(build_messages(question))
    transport = transport or _post_json
    response = transport(base_url + "/chat/completions", payload, timeout)
    return with_disclaimer(extract_answer(response))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(argv).strip() or DEMO_QUESTION
    print("Model : " + MODEL_ID)
    print("Base  : " + DEFAULT_BASE_URL)
    print("Q     : " + question)
    print("")
    try:
        print(chat(question))
    except Exception as exc:
        print("[error] " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
