"""Minimal health-education chatbot demo built on Spark-X2.6-4B.

Goal: make general health knowledge easier to reach in plain language, without
pretending to be a clinician. The model explains, the user decides, and a
qualified professional stays in charge of any diagnosis or treatment.

Run the demo (needs a local OpenAI-compatible Spark-X2.6-4B endpoint):

    python src/health_chatbot.py
    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py "..."

See README.md and FAQ.md for model download and runtime notes.
"""

import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.2
MAX_QUESTION_CHARS = 2000

DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
REQUEST_TIMEOUT = int(os.environ.get("SPARK_TIMEOUT", "120"))

SYSTEM_PROMPT = (
    "You are a patient-education assistant. Explain general health information "
    "in plain language (about a 6th-grade reading level), using short paragraphs "
    "and bullet points.\n"
    "Rules you must never break:\n"
    "1. Do not diagnose. Do not present any disease as the user's condition.\n"
    "2. Do not prescribe or recommend medicines, doses, or treatments.\n"
    "3. Do not tell the user to start, stop, or change a prescribed treatment.\n"
    "4. Say clearly which parts need a qualified clinician, and which questions "
    "and documents the user should bring to that visit.\n"
    "5. If the message suggests an emergency, tell the user to contact local "
    "emergency services immediately.\n"
    "6. Do not ask for or repeat names, addresses, ID numbers, or other personal "
    "identifiers.\n"
    "7. If you are unsure, say so instead of guessing."
)

EMERGENCY_KEYWORDS = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "trouble breathing",
    "severe bleeding",
    "unconscious",
    "seizure",
    "stiff neck",
    "stroke",
    "suicidal",
    "suicide",
)

EMERGENCY_NOTICE = (
    "If this may be an emergency, contact your local emergency number now "
    "instead of waiting for an answer here."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago. I now have a dry cough, "
    "a fever of 38.2 C, and I feel very tired. What should I know before I see "
    "a doctor, and when should I get help right away?"
)


class SparkRequestError(RuntimeError):
    """Raised when the Spark endpoint cannot be reached or returns bad data."""


def check_question(question):
    """Validate and normalise a user question."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    text = " ".join(question.split())
    if not text:
        raise ValueError("question must not be empty")
    if len(text) > MAX_QUESTION_CHARS:
        raise ValueError(
            "question is too long (max %d characters)" % MAX_QUESTION_CHARS
        )
    return text


def needs_emergency_notice(question):
    """True when the text contains a phrase that may indicate an emergency."""
    lowered = question.lower()
    return any(term in lowered for term in EMERGENCY_KEYWORDS)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": check_question(question)})
    return messages


def build_payload(question, history=None):
    """Build the OpenAI-compatible request body."""
    return {
        "model": MODEL_ID,
        "messages": build_messages(question, history),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    }


def extract_answer(response):
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise SparkRequestError("response did not contain choices[0].message")
    content = (message.get("content") or "").strip()
    if not content:
        raise SparkRequestError("model returned an empty answer")
    return content


def format_answer(question, answer):
    """Add the emergency banner and the clinician note to a model answer."""
    parts = []
    if needs_emergency_notice(question):
        parts.append(EMERGENCY_NOTICE)
    parts.append(answer.strip())
    parts.append("This is general education, not medical advice or a diagnosis.")
    return "\n\n".join(parts)


def chat(question, base_url=None, history=None, timeout=None):
    """Send one question to a local Spark-X2.6-4B server and return the answer."""
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    body = json.dumps(build_payload(question, history)).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout or REQUEST_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise SparkRequestError(
            "could not reach %s (%s). Start a local Spark-X2.6-4B server "
            "(see README.md) or set SPARK_BASE_URL." % (url, exc)
        )
    try:
        data = json.loads(raw)
    except ValueError:
        raise SparkRequestError("endpoint returned non-JSON output")
    return format_answer(question, extract_answer(data))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    question = " ".join(args).strip() or DEMO_QUESTION
    try:
        print(chat(question))
    except (ValueError, TypeError, SparkRequestError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
